import heapq
from copy import deepcopy

from path_finding import a_star, construct_heights

"""
CBS to handle construction tasks

Assumptions
    1. The environment meets the 'ring' well-formed condition
        1) The number of tasks is finite
        2) # parking endpoints (non-task endpoints, border) >= # agents
        3) The parking endpoints form a ring
    2. # tasks = # agents
    3. There exists a single-agent path to finish all tasks (from high-level plan)
    4. Do not add and remove the same block (so goals at the same x-y location are all adds or all removes)

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
    0: Flow time = Σ time to finish tasks, ignoring dummy paths
    1: Makespan = max time to finish tasks, ignoring dummy paths
    2: Makespan = min time to finish tasks (early rounds) and max time to return to parking location (last round)

Conflict detect order
    0: default, order = time, conflict type (edge-block, agent-block, move-height, block-height, vertex, edge)
    1: block first, order = (edge-block, agent-block, move-height, block-height), (vertex, edge, block-block), 
       each group sorted by time

Conflict resolve order (several conflicts, each between 2 paths, choose which one to resolve)
    0: default order, use agent index
    1: conflict time order, solve conflict happening earlier first
    2: constraint time order, solve conflict that produces earlier constraints first
    3: constraint type order, (edge-block, agent-block, move-height, block-height), (vertex, edge, block-block)
"""

def insert_stays(goal_info, paths, times, goal=None):
    """Insert stay actions to make all paths satisfy dependency"""
    if goal is None:
        lv = 1
        while lv in goal_info['dep_lv_2_g']:
            goals = goal_info['dep_lv_2_g'][lv]
            for g in goals:
                gid = goal_info['id'][g]
                t_g = times[gid]
                pred = goal_info['pred'][g]
                t_pred = max(times[goal_info['id'][p]] for p in pred)
                if t_pred >= t_g:
                    path = paths[gid]
                    stay = (path[t_g][0], 'move')
                    paths[gid] = path[:t_g] + [stay] * (t_pred - t_g + 1) + path[t_g:]
                    times[gid] = t_pred + 1
            lv += 1
    else:
        affected_goals = [goal] if goal[0] > 0 else []
        while len(affected_goals) > 0:
            next_affected_goals = []
            for g in affected_goals:
                gid = goal_info['id'][g]
                t_g = times[gid]
                for succ in goal_info['succ'][g]:
                    sid = goal_info['id'][succ]
                    t_s = times[sid]
                    if t_g >= t_s:
                        path = paths[sid]
                        stay = (path[t_s][0], 'move')
                        paths[sid] = path[:t_s] + [stay] * (t_g - t_s + 1) + path[t_s:]
                        times[sid] = t_g + 1
                        next_affected_goals.append(succ)
            affected_goals = next_affected_goals
    return paths, times

def extend_paths(paths, window):
    """Extend paths to a fixed length by appending stay actions"""
    for path in paths:
        pos = (path[-1][0], 'move')
        path += [pos] * (window - len(path))


'''Conflict handling'''
def detect_all_conflicts(height, paths, block_actions, detect_mode, priority):
    """Detect conflicts between all pairs of paths"""
    heights, t2hid = construct_heights(height, block_actions)
    conflicts = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            c = detect_conflict(heights, t2hid, paths[i], paths[j], block_actions[i], block_actions[j], detect_mode)
            if c:
                c['a1'] = i
                c['a2'] = j
                c['priority'] = priority[c['type']]
                conflicts.append(c)
    return conflicts

def detect_conflict(heights, t2hid, path1, path2, block_action1, block_action2, mode):
    """
    Detect conflicts between two paths
    Detect order
        0: time, conflict type (edge-block, agent-block, move-height, block-height, vertex, edge)
        1: block first, order = (edge-block, agent-block, move-height, block-height), (vertex, edge, block-block),
           each group sorted by time
    """
    t1, (add1, gx1, gy1, lv1, _) = block_action1
    t2, (add2, gx2, gy2, lv2, _) = block_action2
    gloc1, gloc2 = (gx1, gy1), (gx2, gy2)

    for r in range(2):
        if (r == 1 and mode == 0):
            break

        ploc1, ploc2 = path1[0][0][:2], path2[0][0][:2]
        for t in range(1, len(path1)):
            height = heights[t2hid[t]] if t in t2hid else heights[-1]
            loc1, loc2 = path1[t][0][:2], path2[t][0][:2]

            if r == 0:
                '''Edge-block conflict: a1 moves from A to B, while a2 performs block action from B to A'''
                if t == t1 and gloc1 == ploc2 and loc1 == loc2:
                    return {'type': 'edge-block', 'time': t, 'loc': (loc1, gloc1), 'block': 1, 't': t}
                if t == t2 and gloc2 == ploc1 and loc1 == loc2:
                    return {'type': 'edge-block', 'time': t, 'loc': (loc2, gloc2), 'block': 2, 't': t}

                '''Agent-block conflict: a1 moves from A to B, while a2 performs block action at B'''
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

                '''Move height conflict: a1 moves from A to B, B is a2's goal location, with height difference > 1'''
                h1, h2 = height[loc1], height[loc2]
                ph1, ph2 = height[ploc1], height[ploc2]
                if gloc1 == loc2 and ((ph2 < h2 and lv1 == h2 + 1) or (ph2 > h2 and lv1 == h2 - 2)):
                    return {'type': 'move-height', 'time': t, 'loc': (loc1, loc2), 'block': 1, 't': t}
                if gloc2 == loc1 and ((ph1 < h1 and lv2 == h1 + 1) or (ph1 > h1 and lv2 == h1 - 2)):
                    return {'type': 'move-height', 'time': t, 'loc': (loc2, loc1), 'block': 2, 't': t}

                '''Block height conflict: a1 performs block action from A to B, A = g2, while height is incorrect'''
                if t == t1 and loc1 == gloc2 and lv1 != h1 and (1 >= lv1 - lv2 >= 0):
                    return {'type': 'block-height', 'time': t, 'loc': loc1, 'curr': 1, 't': t, 't2': t2}
                if t == t2 and loc2 == gloc1 and lv2 != h2 and (1 >= lv2 - lv1 >= 0):
                    return {'type': 'block-height', 'time': t, 'loc': loc2, 'curr': 2, 't': t, 't2': t1}

            if (r == 0 and mode == 0) or (r == 1 and mode == 1):
                '''No conflict outside the world'''
                if ploc1[0] == -1 or ploc2[0] == -1 or loc1[0] == -1 or loc2[0] == -1:
                    continue
                '''Vertex conflict'''
                if loc1 == loc2 and loc1[0] != -1:
                    return {'type': 'vertex', 'time': t, 'loc': loc1, 't': t}
                '''Edge conflict'''
                if loc1 == ploc2 and loc2 == ploc1 and loc1[0]:
                    return {'type': 'edge', 'time': t, 'loc': (ploc1, loc1), 't': t}
                '''Block-block conflict: ignored, won't happen'''

            ploc1, ploc2 = loc1, loc2

    return None


def resolve_conflict(conflict):
    """
    Resolve a conflict by adding constraints
    Constraint = (type, agent, time, loc, range), loc and range are optional
    """
    cons1, cons2 = [], []
    loc = conflict['loc']
    time = conflict['time']
    # Vertex conflict
    if conflict['type'] == 'vertex':
        cons1.append(('vertex', conflict['a1'], time, loc))
        cons2.append(('vertex', conflict['a2'], time, loc))
    # Edge conflict
    elif conflict['type'] == 'edge':
        cons1.append(('edge', conflict['a1'], time, loc))
        loc2 = (loc[1], loc[0])
        cons2.append(('edge', conflict['a2'], time, loc2))
    # Agent-block conflict
    elif conflict['type'] == 'agent-block':  # Disallow move or disallow block action
        block_a = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_a = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        cons1.append(('block', block_a, time))
        cons2.append(('vertex', move_a, time, loc))
        # Stay: move agent arrives at t-1 at the latest, leaves at t+1 at the earliest
        if conflict['move'] == 'stay':
            cons1.append(('block', block_a, time - 1))  # Allow move agent to arrive
            cons1.append(('block', block_a, time + 1))  # Allow move agent to leave
            cons2.append(('vertex', move_a, time - 1, loc))  # Disallow arrival
        # Arrive: move agent can leave at t+1 at the earliest
        elif conflict['move'] == 'arrive':
            cons1.append(('block', block_a, time + 1))  # Allow move agent to leave
        # Leave: move agent arrives at t-1 at the latest
        else:
            cons1.append(('block', block_a, time - 1))  # Allow move agent to arrive
            cons2.append(('vertex', move_a, time - 1, loc))  # Disallow arrival
    # Edge-block conflict
    elif conflict['type'] == 'edge-block':
        block_a = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_a = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        cons1.append(('vertex', block_a, time, loc[0]))
        for t in range(time - 1, time + 1):
            cons1.append(('block', block_a, t))
            cons2.append(('vertex', move_a, t, loc[0]))
            cons2.append(('vertex', move_a, t, loc[1]))
    # Move height conflict
    elif conflict['type'] == 'move-height':
        block_a = conflict['a1'] if conflict['block'] == 1 else conflict['a2']
        move_a = conflict['a1'] if conflict['block'] == 2 else conflict['a2']
        tb = conflict['block_t']
        # Block action has not been executed: cannot use this edge until execution
        if time < tb:
            for t in range(time, tb + 1):
                cons1.append(('edge', move_a, time, loc))
        # Block action has been executed: cannot use this edge after execution TODO: forever / until a counter action?
        else:
            cons1.append(('range-edge', move_a, tb, loc))
            for t in range(tb, time + 1):
                # cons1.append(('edge', move_a, t, loc))
                cons2.append(('block', block_a, t))
            cons2.append(('block', block_a, time + 1))  # 1 extra step to leave
    # Block height conflict
    elif conflict['type'] == 'block-height':
        block_curr = conflict['a1'] if conflict['curr'] == 1 else conflict['a2']
        block_nbr = conflict['a1'] if conflict['curr'] == 2 else conflict['a2']
        tn = conflict['t2']
        # Block action at neighbor location has not been executed: cannot use this edge until execution
        if time < tn:
            for t in range(time, tn + 2):  # One extra step to enter
                cons1.append(('block-nbr', block_curr, t, loc))
        # Block action at neighbor location has been executed: cannot use this edge after execution
        else:
            cons1.append(('range-block-nbr', block_curr, tn, loc))
            for t in range(tn, time + 1):
                # cons1.append(('block-nbr', block_curr, t, loc))
                cons2.append(('block', block_nbr, t))
            cons2.append(('block', block_nbr, time + 1))  # 1 extra step to leave
            cons2.append(('block', block_nbr, time + 2))  # 1 extra step to leave
    else:
        raise Exception('Invalid conflict type')
    return cons1, cons2

def compute_cost(paths, block_actions, mode, last_round):
    """
    Compute the cost of a sequence of block actions
    Mode 0: cost = flow time = sum of time steps until goal (do not count dummy path)
    Mode 1: cost = makespan = max of time steps until goal (do not count dummy path)
    Mode 2: cost = makespan = min of time steps until goal, except for last round (use mode 1)
    Mode 3: cost = (mode 2, mode 1)
    """
    if mode == 0:
        cost = 0
        for t, _ in block_actions:
            cost += t
    elif mode == 1:
        cost = 0
        for t, _ in block_actions:
            cost = max(cost, t)
    elif mode == 2:
        if last_round:
            cost = 0
            for p in paths:
                cost = max(cost, len(p))
        else:
            cost = float('inf')
            for t, _ in block_actions:
                if t > 0:
                    cost = min(cost, t)
    else:
        raise NotImplementedError
    return cost

def order_conflicts(conflicts, mode):
    """
    Order conflicts between paths
    Mode 0: default order, use agent index
    Mode 1: conflict time order, solve conflicts happening earlier first
    Mode 2: constraint time order, solve conflicts that produce earlier constraints first
    Mode 3: constraint type order, level - > edge-block -> (vertex, edge, block-block, agent-block)
    """
    if mode == 0:
        pass
    elif mode == 1:
        conflicts.sort(key=lambda x: x['time'])
    elif mode == 2:
        conflicts.sort(key=lambda x: x['t'])
    elif mode >= 3:
        conflicts.sort(key=lambda x: x['priority'])
    else:
        raise NotImplementedError

def push_node(open_list, node):
    """Push a node into the open list. Order = cost, # conflicts, gen_id"""
    heapq.heappush(open_list, (node.cost, len(node.conflicts), node.gen_id, node))


def cbs(env, goal_info, arg, last_round):
    height = env.height
    num = len(goal_info['goals'])
    limit = env.w * env.w

    '''Plan initial single-agent paths'''
    paths, times = [], []
    for i in range(num):
        path, t = a_star(env, goal_info, [], i, arg)
        paths.append(path)
        times.append(t)
    paths, times = insert_stays(goal_info, paths, times)
    window = max([len(p) for p in paths])
    extend_paths(paths, window)
    block_actions = [(times[i], goal_info['goals'][i]) for i in range(num)]
    root = Node(paths, times, block_actions, set(), 0)
    root.cost = compute_cost(paths, block_actions, arg.cost, last_round)
    root.conflicts = detect_all_conflicts(height, paths, block_actions, arg.detect, arg.priority)

    open_list = []
    closed_list = dict()
    generate = expand = dup = 0
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        expand += 1

        '''Completion check'''
        if len(node.conflicts) == 0:
            print(f'Generate: {generate}, Expand: {expand}, Duplicate: {dup}')
            return node.paths, node.times, (generate, expand)

        '''Resolve a conflict'''
        order_conflicts(node.conflicts, arg.resolve)
        conflict = node.conflicts[0]
        constraints = resolve_conflict(conflict)

        '''Generate new nodes'''
        for cons in constraints:
            new_cons = set(cons) - set(node.constraints)  # New constraints not in parent node
            if len(new_cons) == 0:
                continue
            child = Node(deepcopy(node.paths), node.times.copy(), node.block_actions,
                         node.constraints.copy().union(new_cons), generate + 1)
            aid = cons[0][1]
            goal = goal_info['goals'][aid]
            pred = goal_info['pred'][goal]
            earliest = 0 if len(pred) == 0 else max(node.times[goal_info['id'][p]] for p in pred) + 1
            path, t = a_star(env, goal_info, child.constraints, aid, arg, earliest=earliest, latest=limit, paths=child.paths, block_actions=child.block_actions)
            if path:
                child.paths[aid] = path
                child.times[aid] = t
                child.paths, child.times = insert_stays(goal_info, child.paths, child.times, goal_info['goals'][aid])
                window = max([len(p) for p in child.paths])
                extend_paths(child.paths, window)
                child.block_actions = [(child.times[i], goal_info['goals'][i]) for i in range(num)]
                child.cost = compute_cost(child.paths, child.block_actions, arg.cost, last_round)
                child.conflicts = detect_all_conflicts(height, child.paths, child.block_actions, arg.detect, arg.priority)

                # TODO: duplicate detection
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

