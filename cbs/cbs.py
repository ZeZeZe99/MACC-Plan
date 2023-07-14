import numpy as np
import heapq
from copy import deepcopy
import cProfile
import pstats

from cbs.path_finding import process_goal, a_star
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
            heights[hid:, x, y] -= 1
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
            c = detect_conflict(goal_info['loc2goal'], heights, t2hid, paths, block_actions, i, j)
            if c:
                c['a1'] = i
                c['a2'] = j
                conflicts.append(c)
    return conflicts


def detect_conflict(loc2goal, heights, t2hid, paths, block_actions, i, j):
    """Detect conflicts between two paths"""
    t1, (add1, gx1, gy1, lv1, _) = block_actions[i]
    t2, (add2, gx2, gy2, lv2, _) = block_actions[j]
    h1 = heights[t2hid[t1 + 1], gx1, gy1] if t1 > 0 else 0
    h2 = heights[t2hid[t2 + 1], gx2, gy2] if t2 > 0 else 0
    ph1 = heights[t2hid[t1], gx1, gy1] if t1 > 0 else 0
    ph2 = heights[t2hid[t2], gx2, gy2] if t2 > 0 else 0
    gloc1, gloc2 = (gx1, gy1), (gx2, gy2)
    ploc1, ploc2 = paths[i][0][:2], paths[j][0][:2]
    prev_height = heights[0]

    for t in range(1, len(paths[i])):
        height = heights[t2hid[t]] if t in t2hid else heights[-1]
        loc1, loc2 = paths[i][t][:2], paths[j][t][:2]
        # TODO: detect order (e.g. vertex vs. agent-block)
        # Vertex conflict
        if loc1 == loc2:
            return {'type': 'vertex', 'time': t, 'loc': loc1}
        # Edge conflict
        if loc1 == ploc2 and loc2 == ploc1:
            return {'type': 'edge', 'time': t, 'loc': (ploc1, loc1)}

        # Agent-block conflict
        if t == t1:
            if gloc1 == ploc2 == loc2:  # Agent 2 staying at agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'stay'}
            if gloc1 == ploc2:  # Agent 2 leaving agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'leave'}
            if gloc1 == loc2:  # Agent 2 arriving at agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'arrive'}
        if t == t2:
            if gloc2 == ploc1 == loc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'stay'}
            if gloc2 == ploc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'leave'}
            if gloc2 == loc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'arrive'}

        # Block-block conflict
        if t == t1 and t == t2 and gloc1 == gloc2:
            return {'type': 'block-block', 'time': t, 'loc': gloc1}

        # Block-map conflict (a block action is invalid due to goal neighbor location has incorrect height)
        if t == t1 and loc1 == gloc2:  # Agent 1 performs block action standing on agent 2's goal location
            # Agent 2 has executed and leads to incorrect height / hasn't executed but can make height correct
            if (t > t2 and ph2 == lv1) or (t < t2 and h2 == lv1):
                return {'type': 'block-map', 'time': t, 'loc': loc1, 'neighbor': 2, 'neighbor_t': t2}
        if t == t2 and loc2 == gloc1:
            if (t > t1 and ph1 == lv2) or (t < t1 and h1 == lv2):
                return {'type': 'block-map', 'time': t, 'loc': loc2, 'neighbor': 1, 'neighbor_t': t1}

        # Agent-map conflict (an agent's block action invalidates another agent's move action)
        if abs(prev_height[ploc1[0], ploc1[1]] - height[loc1[0], loc1[1]]) > 1:
            if gloc2 != ploc1 and gloc2 != loc1:
                continue
            if gloc2 == ploc1:  # Agent 1 leaving agent 2's block location
                h = height[loc1[0], loc1[1]]  # h = height of the other location
                arrive = False
                loc = loc1
            else:  # Agent 1 arriving at agent 2's block location
                h = prev_height[ploc1[0], ploc1[1]]
                arrive = True
                loc = ploc1
            # Agent 2's block action should lead to height difference > 1
            if not (t > t2 and abs(h2 - h) > 1 >= abs(ph2 - h)) and not (t < t2 and abs(ph2 - h) > 1 >= abs(h2 - h)):
                continue
            if loc not in loc2goal:  # Agent 2's block action is the sole reason
                return {'type': 'agent-map', 'time': t, 'loc': (ploc1, loc1),
                        'block': 2, 'block_t': t2, 'arrive': arrive}
            # A third agent may be involved
            for g in loc2goal[loc]:
                add, gx3, gy3, lv, k = g
                if k == i or k == j:
                    continue
                t3 = block_actions[k][0]
                if t > t3:  # Agent 3 has executed its block action
                    ph3 = heights[t2hid[t3], gx3, gy3]
                    h3 = h
                else:  # Agent 3 has not executed its block action
                    ph3 = h
                    h3 = heights[t2hid[t3 + 1], gx3, gy3]
                if (t > t2 and t > t3 and abs(h2 - h3) > 1 >= abs(h2 - ph3)) or \
                        (t2 > t > t3 and abs(ph2 - h3) > 1 >= abs(ph2 - ph3)) or \
                        (t2 < t < t3 and abs(h2 - ph3) > 1 >= abs(h2 - h3)) or \
                        (t < t2 and t < t3 and abs(ph2 - ph3) > 1 >= abs(ph2 - h3)):
                    return {'type': 'agent-map', 'time': t, 'loc': (ploc1, loc1),
                            'block': 2, 'block_t': t2, 'arrive': arrive, 'third': k, 'third_t': t3}
        if abs(prev_height[ploc2[0], ploc2[1]] - height[loc2[0], loc2[1]]) > 1:
            if gloc1 != ploc2 and gloc1 != loc2:
                continue
            if gloc1 == ploc2:
                h = height[loc2[0], loc2[1]]
                arrive = False
                loc = loc2
            else:
                h = prev_height[ploc2[0], ploc2[1]]
                arrive = True
                loc = ploc2
            if not (t > t1 and abs(h1 - h) > 1 >= abs(ph1 - h)) and not (t < t1 and abs(ph1 - h) > 1 >= abs(h1 - h)):
                continue
            if loc not in loc2goal:
                return {'type': 'agent-map', 'time': t, 'loc': (ploc2, loc2),
                        'block': 1, 'block_t': t1, 'arrive': arrive}
            for g in loc2goal[loc]:
                add, gx3, gy3, lv, k = g
                if k == i or k == j:
                    continue
                t3 = block_actions[k][0]
                if t > t3:
                    ph3 = heights[t2hid[t3], gx3, gy3]
                    h3 = h
                else:
                    ph3 = h
                    h3 = heights[t2hid[t3 + 1], gx3, gy3]
                if (t > t1 and t > t3 and abs(h1 - h3) > 1 >= abs(h1 - ph3)) or \
                        (t1 > t > t3 and abs(ph1 - h3) > 1 >= abs(ph1 - ph3)) or \
                        (t1 < t < t3 and abs(h1 - ph3) > 1 >= abs(h1 - h3)) or \
                        (t < t1 and t < t3 and abs(ph1 - ph3) > 1 >= abs(ph1 - h3)):
                    return {'type': 'agent-map', 'time': t, 'loc': (ploc2, loc2),
                            'block': 1, 'block_t': t1, 'arrive': arrive, 'third': k, 'third_t': t3}


        ploc1, ploc2 = loc1, loc2
        prev_height = height

    return None


def resolve_conflict(conflict):
    """Resolve a conflict by adding constraints"""
    cons1, cons2, cons3 = [], [], []
    loc = conflict['loc']
    time = conflict['time']
    if conflict['type'] == 'vertex':
        cons1.append({'type': 'vertex', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        cons2.append({'type': 'vertex', 'agent': conflict['a2'], 'loc': loc, 'time': time})
    elif conflict['type'] == 'edge':
        cons1.append({'type': 'edge', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        loc2 = (loc[1], loc[0])
        cons2.append({'type': 'edge', 'agent': conflict['a2'], 'loc': loc2, 'time': time})
    elif conflict['type'] == 'agent-block':  # Disallow move or disallow block action
        block_agent = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_agent = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        cons1.append({'type': 'block', 'agent': block_agent, 'time': time})
        cons2.append({'type': 'vertex', 'agent': move_agent, 'time': time, 'loc': loc})
        if conflict['move'] == 'stay':  # Move agent arrives at t-1 at the latest, leaves at t+1 at the earliest
            cons1.append({'type': 'block', 'agent': block_agent, 'time': time + 1})  # Allow move agent to leave
            cons1.append({'type': 'block', 'agent': block_agent, 'time': time - 1})  # Allow move agent to arrive
            cons2.append({'type': 'vertex', 'agent': move_agent, 'time': time - 1, 'loc': loc})  # Disallow arrival
        elif conflict['move'] == 'arrive':  # Move agent will leave at t+1 at the earliest
            cons1.append({'type': 'block', 'agent': block_agent, 'time': time + 1})  # Allow move agent to leave
        else:  # Move agent arrives at t-1 at the latest
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
        if 'third' in conflict:
            t3, a3 = conflict['third_t'], conflict['third']
            if time < tb < t3:
                for t in range(tb - arrive + 1, t3 + arrive + 1):
                    cons1.append({'type': 'edge', 'agent': move_agent, 'loc': loc, 'time': t})
            elif tb > time > t3:
                for t in range(t3, time - arrive + 2):
                    cons2.append({'type': 'block', 'agent': a3, 'time': t})
            elif tb < time < t3:
                cons1.pop(0)
                for t in range(tb - 1 + arrive, t3 + arrive):
                    cons1.append({'type': 'edge', 'agent': move_agent, 'loc': loc, 'time': t})
            elif time > tb and time > t3:
                pass



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
    closed_list = set()
    generate = expand = 0
    root = Node(height, paths, times, block_actions, [], generate)
    root.cost = compute_cost(block_actions)
    root.conflicts = detect_all_conflicts(goal_info, height, paths, block_actions)
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        expand += 1
        # print(expand)

        '''Completion check'''
        if len(node.conflicts) == 0:
            print(f'Generate: {generate}, Expand: {expand}')
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
                key = (str(map(str, child.paths)), str(child.constraints))
                if key not in closed_list:
                    closed_list.add(key)
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
    positions = [(1, 3), (1, 1), (1, 2)]
    goals = [(0, 2, 2, 1), (1, 3, 2, 2), (0, 1, 2, 0)]
    carry_stats = [False, True, False]

    profiler = cProfile.Profile()
    profiler.enable()
    paths, times = cbs(env, goals, positions, carry_stats)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    for p in paths:
        print(p)
