import numpy as np
import heapq
from copy import deepcopy
import cProfile
import pstats

from path_finding import process_goal, a_star
import lego
import config

"""
CBS to handle construction tasks

Assumptions
    1. The environment meets the 'ring' well-formed condition
        1) The number of tasks is finite
        2) # parking endpoints (non-task endpoints, border) >= # agents
        3) The parking endpoints form a ring
    2. # tasks = # agents
    3. There exists a single-agent path to finish all tasks (from high-level plan)

Properties
    1. All instances are solvable
    2. CBS is complete and optimal
    3. Can handle tasks with inter-dependencies, including nested dependencies

Approach
    1. Similar to the original CBS
    2. Construct a height map sequence from all paths in a CT node
    3. Extra conflicts
        1) Block-block conflict: two block actions at the same x-y location at the same time
        2) Agent-block conflict: a block action performed at the same x-y location the other agent is using
        3) Level conflict: z level of an agent's path does not match the current height map
    4. Modified low-level search
        1) Search in 3D space. z level is determined by possible heights affected by all tasks
        2) Path = agent location (-> depo) -> goal location -> border
    5. Dummy path
        1) Path back to border, after finishing the task
        2) Must plan, and must avoid collisions with the dummy path
        3) Guarantee that agents won't get stuck
        4) Always exists, from the high-level plan

Cost mode
    0: Flow time = Σ time to finish all tasks, ignoring dummy paths

Conflict resolve order (choosing conflict between which two paths)
    0: default order, use agent index
    1: conflict time order, solve conflicts happening earlier first
    2: constraint time order, solve conflicts that produce earlier constraints first
    
"""


cost_mode = 0
resolve_order = 1
conflict_priority = 0


def insert_stays(goal_info, paths, times, loc=None):
    """Insert stay actions to make all paths 'in-order' (meet the goal ordering)"""
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
                stay = (path[t2][0], 'move')
                paths[gid2] = path[:t2] + [stay] * (t1 - t2 + 1) + path[t2:]
                times[gid2] = t1 + 1
    return paths, times

def extend_paths(paths, window):
    """Extend all paths to a fixed length by appending stay actions"""
    for path in paths:
        pos = (path[-1][0], 'move')
        path += [pos] * (window - len(path))
    return paths


'''Conflict handling'''
def construct_heights(height, block_actions):
    """
    Construct the height map sequence from block actions, plus a mapping from t to height map id
    Use for conflict detection
    Block action: (t, (add, x, y, lv, gid))
    Assume block action at t will change the map at t+1
    """
    block_actions = block_actions.copy()
    block_actions.sort()  # Sort by time
    unique = set([b[0] for b in block_actions])  # Unique time steps
    unique.discard(-1)  # Ignore dummy action
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

def detect_all_conflicts(height, paths, block_actions):
    """Detect conflicts between all pairs of paths"""
    heights, t2hid = construct_heights(height, block_actions)
    conflicts = []
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
    gloc1, gloc2 = (gx1, gy1), (gx2, gy2)
    ploc1, ploc2 = path1[0][0][:2], path2[0][0][:2]

    for t in range(1, len(path1)):
        height = heights[t2hid[t]] if t in t2hid else heights[-1]
        loc1, loc2 = path1[t][0][:2], path2[t][0][:2]
        z1, z2 = path1[t][0][2], path2[t][0][2]

        # Vertex conflict
        if loc1 == loc2:
            return {'type': 'vertex', 'time': t, 'loc': loc1, 't': t}

        # Edge conflict
        if loc1 == ploc2 and loc2 == ploc1:
            return {'type': 'edge', 'time': t, 'loc': (ploc1, loc1), 't': t}

        # Block-block conflict
        if t == t1 == t2 and gloc1 == gloc2:
            return {'type': 'block-block', 'time': t, 'loc': gloc1, 't': t}

        # Agent-block conflict
        if t == t1:
            if gloc1 == ploc2 == loc2:  # Agent 2 staying at agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'stay', 't': t}
            if gloc1 == ploc2:  # Agent 2 leaving agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'leave', 't': t}
            if gloc1 == loc2:  # Agent 2 arriving at agent 1's block location
                return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'arrive', 't': t}
        if t == t2:
            if gloc2 == ploc1 == loc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'stay', 't': t}
            if gloc2 == ploc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'leave', 't': t}
            if gloc2 == loc1:
                return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'arrive', 't': t}

        # Level conflict (agent's z level doesn't match the height, due to the other agent's block action)
        if gloc1 == loc2 and height[loc2] != z2 and (lv1 == z2 or lv1 == z2 - 1):
            return {'type': 'level', 'time': t, 'loc': (loc2, z2), 'block': 1, 'block_t': t1, 't': min(t, t1)}
        if gloc2 == loc1 and height[loc1] != z1 and (lv2 == z1 or lv2 == z1 - 1):
            return {'type': 'level', 'time': t, 'loc': (loc1, z1), 'block': 2, 'block_t': t2, 't': min(t, t2)}

        ploc1, ploc2= loc1, loc2
    return None

def resolve_conflict(conflict):
    """Resolve a conflict by adding constraints"""
    cons1, cons2 = [], []
    loc = conflict['loc']
    time = conflict['time']
    # Vertex conflict
    if conflict['type'] == 'vertex':
        cons1.append({'type': 'vertex', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        cons2.append({'type': 'vertex', 'agent': conflict['a2'], 'loc': loc, 'time': time})
    # Edge conflict
    elif conflict['type'] == 'edge':
        cons1.append({'type': 'edge', 'agent': conflict['a1'], 'loc': loc, 'time': time})
        loc2 = (loc[1], loc[0])
        cons2.append({'type': 'edge', 'agent': conflict['a2'], 'loc': loc2, 'time': time})
    # Block-block conflict
    elif conflict['type'] == 'block-block':
        cons1.append({'type': 'block', 'agent': conflict['a1'], 'time': time})
        cons2.append({'type': 'block', 'agent': conflict['a2'], 'time': time})
    # Agent-block conflict
    elif conflict['type'] == 'agent-block':
        block_a = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_a = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        cons1.append({'type': 'block', 'agent': block_a, 'time': time})
        cons2.append({'type': 'vertex', 'agent': move_a, 'time': time, 'loc': loc})
        # Stay: move agent arrives at t-1 at the latest, leaves at t+1 at the earliest
        if conflict['move'] == 'stay':
            cons1.append({'type': 'block', 'agent': block_a, 'time': time - 1})  # Allow move agent to arrive
            cons1.append({'type': 'block', 'agent': block_a, 'time': time + 1})  # Allow move agent to leave
            cons2.append({'type': 'vertex', 'agent': move_a, 'time': time - 1, 'loc': loc})  # Disallow arrival
        # Arrive: move agent can leave at t+1 at the earliest
        elif conflict['move'] == 'arrive':
            cons1.append({'type': 'block', 'agent': block_a, 'time': time + 1})  # Allow move agent to leave
        # Leave: move agent arrives at t-1 at the latest
        else:
            cons1.append({'type': 'block', 'agent': block_a, 'time': time - 1})  # Allow move agent to arrive
            cons2.append({'type': 'vertex', 'agent': move_a, 'time': time - 1, 'loc': loc})  # Disallow arrival
    # Level conflict
    elif conflict['type'] == 'level':
        block_a = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_a = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        tb = conflict['block_t']
        # Block action has not been executed: cannot use this level vertex until execution
        if time < tb:
            for t in range(time, tb + 1):
                cons1.append({'type': 'lv-vertex', 'agent': move_a, 'time': t, 'loc': loc, 'range': False})
        # Block action has been executed: cannot use this level vertex after execution
        else:
            cons1.append({'type': 'lv-vertex', 'agent': move_a, 'time': tb, 'loc': loc, 'range': True})
            for t in range(tb, time + 2):  # 1 extra step to leave
                cons2.append({'type': 'block', 'agent': block_a, 'time': t})
    else:
        raise Exception('Invalid conflict type')
    return cons1, cons2

def compute_cost(block_actions):
    """
    Compute the cost of a sequence of block actions
    Mode 0: cost = flow time = sum of time steps until goal (do not count dummy path)
    Mode 1: cost = fuel use = sum of active actions until goal (do not count dummy path, do not count stay action)
    """
    if cost_mode == 0:
        cost = 0
        for t, _ in block_actions:
            cost += t
    else:
        raise NotImplementedError
    return cost

def order_conflicts(conflicts):
    """
    Order conflicts between paths
    Mode 0: default order, use agent index
    Mode 1: conflict time order, solve conflicts happening earlier first
    Mode 2: constraint time order, solve conflicts that produce earlier constraints first
    """
    if resolve_order == 0:
        pass
    elif resolve_order == 1:
        conflicts.sort(key=lambda x: x['time'])
    elif resolve_order == 2:
        conflicts.sort(key=lambda x: x['t'])

def push_node(open_list, node):
    """Push a node into the open list. Order = cost, # conflicts, gen_id"""
    heapq.heappush(open_list, (node.cost, len(node.conflicts), node.gen_id, node))

def cbs(env, goals, positions, carry_stats):
    height = env.height
    goal_info = process_goal(env, goals.copy(), carry_stats)
    num = len(goals)
    limit = env.w * env.w

    '''Plan initial single-agent paths'''
    paths, times = [], []
    for i in range(len(positions)):
        path, t = a_star(env, goal_info, positions, [], i)
        paths.append(path)
        times.append(t)
    paths, times = insert_stays(goal_info, paths, times)
    window = max([len(p) for p in paths])
    paths = extend_paths(paths, window)  # Pad all paths to the same length
    block_actions = [(times[i], goal_info['goals'][i]) for i in range(num)]
    root = Node(paths, times, block_actions, [], 0)
    root.cost = compute_cost(block_actions)
    root.conflicts = detect_all_conflicts(height, paths, block_actions)

    open_list = []
    closed_list = set()
    generate = expand = dup = 0
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        expand += 1

        '''Completion check'''
        if len(node.conflicts) == 0:
            print(f'Generate: {generate}, Expand: {expand}, Duplicate: {dup}')
            return node.paths, node.times

        '''Resolve a conflict'''
        order_conflicts(node.conflicts)
        conflict = node.conflicts[0]
        constraints = resolve_conflict(conflict)

        '''Generate new nodes'''
        for cons in constraints:
            if len(cons) == 0 or all(c in node.constraints for c in cons):
                continue
            child = Node(deepcopy(node.paths), node.times.copy(), None,
                         node.constraints.copy() + cons, generate + 1)
            aid = cons[0]['agent']
            path, t = a_star(env, goal_info, positions, child.constraints, aid, limit=limit)
            if path:
                child.paths[aid] = path
                child.times[aid] = t
                loc = goal_info['goals'][aid][1:3]
                child.paths, child.times = insert_stays(goal_info, child.paths, child.times, loc=loc)
                window = max([len(p) for p in child.paths])
                child.paths = extend_paths(child.paths, window)
                child.block_actions = [(child.times[i], goal_info['goals'][i]) for i in range(num)]
                child.cost = compute_cost(child.block_actions)
                child.conflicts = detect_all_conflicts(height, child.paths, child.block_actions)

                # TODO: duplicate detection
                key = str(child.constraints)
                if key in closed_list:
                    dup += 1
                else:
                    closed_list.add(key)
                    generate += 1
                    push_node(open_list, child)

    print(f'No solution found. Generate: {generate}, Expand: {expand}')


class Node:
    def __init__(self, paths, times, block_actions, constraints, gen_id):
        self.paths = paths
        self.times = times
        self.block_actions = block_actions
        self.constraints = constraints
        self.gen_id = gen_id
        self.cost = None
        self.conflicts = None


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    # env.height[1, 2] = 1
    # env.height[2:4, 2] = 2
    # positions = [(0, 1, 0), (1, 2, 1), (1, 5, 0)]
    # goals = [(1, 3, 2, 2), (0, 2, 2, 1), (0, 1, 2, 0)]
    # carry_stats = [True, False, False]

    env.height[2, 3] = 2
    env.height[3, 2] = 2
    env.height[3, 3:5] = 1
    carry_stats = [False, False, False]
    goals = [(0, 3, 3, 0), (1, 3, 4, 1), (1, 4, 2, 0)]
    positions = [(3, 3, 1), (4, 1, 0), (3, 4, 1)]

    # profiler = cProfile.Profile()
    # profiler.enable()
    # paths, times = cbs(env, goals, positions, carry_stats)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()

    paths, times = cbs(env, goals, positions, carry_stats)

    for p in paths:
        print(p)
