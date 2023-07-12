import numpy as np
import heapq
from copy import deepcopy
import cProfile
import pstats

from path_finding import process_goal, a_star
import lego
import config


def construct_heights(goal_info, height, block_actions, ignore=None):
    """Construct the height map sequence from block actions, plus a mapping from t to height map id"""
    # Block actions: (t, (add, x, y, lv, gid))
    block_actions = block_actions.copy()
    if ignore is not None:  # Ignore an agent's action (and all goal actions ordered after it)
        if block_actions[ignore][1][0] != -1:
            loc = block_actions[ignore][1][1:3]
            order = goal_info['id2order'][ignore]
            for g in goal_info['loc2goal'][loc][order:]:
                gid = g[4]
                block_actions[gid] = (-1, g)

    block_actions.sort()  # Sort by time
    unique = set([b[0] for b in block_actions])  # Unique time steps
    unique.discard(-1)
    heights = np.tile(height, (len(unique) + 1, 1, 1))  # Height map sequence, include initial one
    t2hid = {0: 0}
    prev_t, hid = 0, 0

    for t, (add, x, y, lv, _) in block_actions:  # t will always start above 0
        if t < 0:  # Skip ignored actions
            continue
        if t > prev_t:
            hid += 1
        h = heights[hid, x, y]
        # Only update height map if the action can be performed
        if add and lv == h:
            heights[hid:, x, y] += 1
        elif not add and lv == h - 1:
            heights[hid, x, y] -= 1
        # Update mapping from t to height map id
        if t != prev_t:
            for i in range(prev_t, t):
                t2hid[i + 1] = hid - 1
            t2hid[t + 1] = hid
        prev_t = t
    return heights, t2hid

def insert_stays(goal_info, paths, times, loc=None):
    """Insert stay actions to make all paths 'in-order' (meet the goal order)"""
    if loc is None:
        locs = goal_info['loc2goal']
    else:
        locs = [loc]
    for loc in locs:
        if loc == (-1, -1):
            continue
        for i in range(1, len(goal_info['loc2goal'][loc])):
            g1, g2 = goal_info['loc2goal'][loc][i-1: i+1]  # g1 is ordered before g2
            gid1, gid2 = g1[4], g2[4]
            t1, t2 = times[gid1], times[gid2]
            if t1 >= t2:  # Need to insert stay actions to make g2 happen after g1
                path = paths[gid2]
                stay = (path[t2][0], path[t2][1], 'move')
                paths[gid2] = path[:t2] + [stay] * (t1 - t2 + 1) + path[t2:]
                times[gid2] = t1 + 1
    return paths, times


def extend_paths(paths, window):
    """Extend paths to a fixed length by appending stay actions"""
    for path in paths:
        pos = (path[-1][0], path[-1][1], 'move')
        path += [pos] * (window - len(path))
    return paths


def detect_all_conflicts(goal_info, height, paths, block_actions):
    """Detect conflicts between all pairs of paths"""
    heights, t2hid = construct_heights(goal_info, height, block_actions)
    conflicts = []
    # TODO: can we stop early?
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            c = detect_conflict(heights, t2hid, paths[i], paths[j], block_actions[i], block_actions[j])
            if c:
                c['a1'] = i
                c['a2'] = j
                conflicts.append(c)
    return conflicts


def detect_conflict(heights, t2hid, path1, path2, block_action1, block_action2):
    """Detect conflicts between two paths"""
    t1, (add1, gx1, gy1, lv1, _) = block_action1
    t2, (add2, gx2, gy2, lv2, _) = block_action2

    px1, py1, px2, py2 = path1[0][0], path1[0][1], path2[0][0], path2[0][1]
    prev_height = heights[0]
    for t in range(1, len(path1)):
        height = heights[t2hid[t]] if t in t2hid else heights[-1]
        x1, y1, x2, y2 = path1[t][0], path1[t][1], path2[t][0], path2[t][1]
        # TODO: detect order (e.g. vertex vs. agent-block)
        # Vertex conflict
        if x1 == x2 and y1 == y2:
            return {'type': 'vertex', 'time': t, 'loc': (x1, y1)}
        # Edge conflict
        if x1 == px2 and y1 == py2 and x2 == px1 and y2 == py1:
            return {'type': 'edge', 'time': t, 'loc': (px1, py1, x1, y1)}

        # Agent-block conflict
        if t == t1:
            if gx1 == px2 and gy1 == py2:  # Agent 2 leaving agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': (gx1, gy1), 'block': 1, 'arrive': False}
            if gx1 == x2 and gy1 == y2:  # Agent 2 arriving at agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': (gx1, gy1), 'block': 1, 'arrive': True}
        if t == t2:
            if gx2 == px1 and gy2 == py1:
                return {'type': 'agent-block', 'time': t, 'loc': (gx2, gy2), 'block': 2, 'arrive': False}
            if gx2 == x1 and gy2 == y1:
                return {'type': 'agent-block', 'time': t, 'loc': (gx2, gy2), 'block': 2, 'arrive': True}

        # Block-block conflict
        if t == t1 and t == t2 and gx1 == gx2 and gy1 == gy2:
            return {'type': 'block-block', 'time': t, 'loc': (gx1, gy1)}

        # Agent-map conflict (an agent's block action invalidates another agent's move action)
        if abs(prev_height[px2, py2] - height[x2, y2]) > 1:  # Height difference > 1: invalid move
            if gx1 == px2 and gy1 == py2:
                return {'type': 'agent-map', 'time': t, 'loc': (px2, py2, x2, y2),
                        'block': 1, 'block_t': t1, 'arrive': False}
            elif gx1 == x2 and gy1 == y2:
                return {'type': 'agent-map', 'time': t, 'loc': (px2, py2, x2, y2),
                        'block': 1, 'block_t': t1, 'arrive': True}
        if abs(prev_height[px1, py1] - height[x1, y1]) > 1:
            if gx2 == px1 and gy2 == py1:
                return {'type': 'agent-map', 'time': t, 'loc': (px1, py1, x1, y1),
                        'block': 2, 'block_t': t2, 'arrive': False}
            elif gx2 == x1 and gy2 == y1:
                return {'type': 'agent-map', 'time': t, 'loc': (px1, py1, x1, y1),
                        'block': 2, 'block_t': t2, 'arrive': True}

        # Block-map conflict (a block action is invalid due to goal neighbor location has incorrect height)
        if t == t1 and x1 == gx2 and y1 == gy2 and height[x1, y1] != lv1:
            return {'type': 'block-map', 'time': t, 'loc': (x1, y1), 'neighbor': 2, 'neighbor_t': t2}
        if t == t2 and x2 == gx1 and y2 == gy1 and  height[x2, y2] != lv2:
            return {'type': 'block-map', 'time': t, 'loc': (x2, y2), 'neighbor': 1, 'neighbor_t': t1}

        px1, py1, px2, py2 = x1, y1, x2, y2
        prev_height = height

    return None


def resolve_conflict(conflict):
    """Resolve a conflict by adding constraints"""
    cons1, cons2 = [], []
    loc = conflict['loc']
    time = conflict['time']
    if conflict['type'] == 'vertex':
        cons1.append({'type': 'vertex', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        cons2.append({'type': 'vertex', 'agent': conflict['a2'], 'loc': loc, 'time': time})
    elif conflict['type'] == 'edge':
        cons1.append({'type': 'edge', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        loc2 = (loc[2], loc[3], loc[0], loc[1])
        cons2.append({'type': 'edge', 'agent': conflict['a2'], 'loc': loc2, 'time': time})
    elif conflict['type'] == 'agent-block':  # Disallow move or disallow block action
        block_agent = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_agent = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        cons1.append({'type': 'block', 'agent': block_agent, 'time': time})
        cons2.append({'type': 'vertex', 'agent': move_agent, 'time': time, 'loc': loc})
        if conflict['arrive']:  # Move agent will leave at t + 1 at the earliest
            cons1.append({'type': 'block', 'agent': block_agent, 'time': time + 1})  # Allow move agent to leave
        else:  # Move agent arrives at t - 1 at the latest
            cons1.append({'type': 'block', 'agent': block_agent, 'time': time - 1})  # Allow arrival
            cons2.append({'type': 'vertex', 'agent': move_agent, 'time': time - 1, 'loc': loc})  # Disallow arrival
    elif conflict['type'] == 'block-block':
        cons1.append({'type': 'block', 'agent': conflict['a1'], 'time': time})
        cons2.append({'type': 'block', 'agent': conflict['a2'], 'time': time})
    elif conflict['type'] == 'agent-map':  # Disallow move or disallow block action
        block_agent = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_agent = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        tb, arrive = conflict['block_t'], conflict['arrive']
        if time < tb:  # Block action has not been executed yet
            for t in range(time, tb - arrive + 2):
                cons1.append({'type': 'edge', 'agent': move_agent, 'loc': loc, 'time': t})
        else:  # Block action has been executed
            cons1.append({'type': 'edge2', 'agent': move_agent, 'loc': loc, 'time': tb})  # Disallow edge from time t
            for t in range(time + arrive + 1):
                cons2.append({'type': 'block', 'agent': block_agent, 'time': t})
    elif conflict['type'] == 'block-map':  # Disallow block action at goal location or neighbor location
        neighbor = conflict['a1'] if conflict['neighbor'] == 1 else conflict['a2']
        goal = conflict['a1'] if conflict['neighbor'] == 2 else conflict['a2']
        tn = conflict['neighbor_t']
        if time < tn:  # Neighbor block action has not been executed yet
            for t in range(time, tn + 2):  # Disallow goal block action from neighbor location
                cons1.append({'type': 'block-edge', 'agent': goal, 'loc': loc, 'time': t})
        else:  # Neighbor block action has been executed
            for t in range(tn, time + 2):  # Disallow neighbor block action (1 extra step for leaving)
                cons1.append({'type': 'block', 'agent': neighbor, 'loc': loc, 'time': t})
    else:
        raise Exception('Invalid conflict type')
    return [cons1, cons2]

def compute_cost(block_actions):
    """
    Compute cost of a solution
    Mode 0: cost = sum of time steps until block action (do not count dummy path)
    Mode 1: cost = sum of active actions until block action (do not count dummy path, do not count stay action)
    """
    cost = 0
    for t, _ in block_actions:
        cost += t
    return cost


def push_node(open_list, node):
    """Push a node into the open list. Order = cost, # conflicts, gen_id"""
    heapq.heappush(open_list, (node.cost, len(node.conflicts), node.gen_id, node))


def cbs(env, goals, positions, carry_stats):
    height = env.height
    goal_info = process_goal(goals.copy(), height, carry_stats)
    num = len(goals)
    limit = env.w * env.w

    # Find initial single-agent paths
    heights = [height]
    t2hid = {0: 0}
    paths, times = [], []
    for i in range(len(positions)):
        path, t = a_star(env, goal_info, positions, [], heights, t2hid, i)
        paths.append(path)
        times.append(t)
    paths, times = insert_stays(goal_info, paths, times)
    window = max([len(p) for p in paths])
    paths = extend_paths(paths, window)
    block_actions = [(times[i], goal_info['goals'][i]) for i in range(num)]

    open_list = []
    generate = expand = 0
    root = Node(height, paths, times, block_actions, [], generate)
    root.cost = compute_cost(block_actions)
    root.conflicts = detect_all_conflicts(goal_info, height, paths, block_actions)
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        expand += 1

        '''Completion check'''
        if len(node.conflicts) == 0:
            # print(f'Generate: {generate}, Expand: {expand}')
            return node.paths, node.times

        '''Resolve a conflict'''
        conflict = node.conflicts[0]
        constraints = resolve_conflict(conflict)

        '''Generate new nodes'''
        # TODO: duplicate detection
        for cons in constraints:
            if len(cons) == 0 or all(c in node.constraints for c in cons):
                continue
            child = Node(height, deepcopy(node.paths), node.times.copy(), None,
                         node.constraints.copy() + cons, generate + 1)
            aid = cons[0]['agent']
            heights, t2hid = construct_heights(goal_info, height, node.block_actions, ignore=aid)
            path, t = a_star(env, goal_info, positions, child.constraints, heights, t2hid, aid, limit=limit)
            if path:
                child.paths[aid] = path
                child.times[aid] = t
                loc = goal_info['goals'][aid][1:3]
                child.paths, child.times = insert_stays(goal_info, child.paths, child.times, loc=loc)
                window = max([len(p) for p in child.paths])
                child.paths = extend_paths(child.paths, window)
                child.block_actions = [(child.times[i], goal_info['goals'][i]) for i in range(num)]
                child.cost = compute_cost(child.block_actions)
                child.conflicts = detect_all_conflicts(goal_info, height, child.paths, child.block_actions)
                generate += 1
                push_node(open_list, child)

    print(f'No solution found. Generate: {generate}, Expand: {expand}')


class Node:
    def __init__(self, height, paths, times, block_actions, constraints, gen_id):
        self.height = height
        self.paths = paths
        self.times = times
        self.block_actions = block_actions
        self.constraints = constraints
        self.gen_id = gen_id


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.height[1, 2] = 1
    env.height[2:4, 2] = 2
    positions = [(1, 1), (1, 2)]
    goals = [(1, 3, 2, 2), (0, 2, 2, 1)]
    carry_stats = [True, False]

    profiler = cProfile.Profile()
    profiler.enable()
    paths, times = cbs(env, goals, positions, carry_stats)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    # for p in paths:
    #     print(p)
