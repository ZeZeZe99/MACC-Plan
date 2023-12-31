import heapq
from copy import deepcopy

from path_finding_2d import a_star, construct_heights

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
        3) Move-height conflict: an agent moves to a location when height difference > 1
        4) Block-height conflict: a block action performed at a location when relative height is incorrect
    4. Modified low-level search
        1) Use Multi-Label A*
        2) Path = agent location (-> depo) -> goal location -> parking location
        3) Graph is modified to include an edge only if a height difference <= 1 exists between two neighbors
    5. Dummy path
        1) Path back to parking location, after finishing the task
        2) Must plan, and must avoid collisions with the dummy path
        3) Guarantee that agents won't get stuck
        4) Always exists, from the high-level plan

Cost mode
    0: Flow time = Σ time to finish tasks, ignoring dummy paths
    1: Makespan = max time to finish tasks (early rounds), max time to return to parking location (last round)
    2: Makespan = min time to finish tasks (early rounds), max time to return to parking location (last round)

Conflict detect order
    0: default, order = time, conflict type (edge-block, agent-block, move-height, block-height, vertex, edge)
    1: block first, order = (edge-block, agent-block, move-height, block-height), (vertex, edge, block-block), 
       each group sorted by time

Conflict resolve order (several conflicts, each between 2 paths, choose which one to resolve)
    0: default order, use agent index
    1: conflict time order, solve conflict happening earlier first
    2: constraint time order, solve conflict that produces earlier constraints first
    3: constraint type order, (block, height) -> (vertex, edge)
    4: constraint type order, (block) -> (height) -> (vertex, edge)
    5: constraint type order, (height) -> (block) -> (vertex, edge)
"""

def insert_stays(goal_info, paths, times, goal=None):
    """Insert stay actions to make all paths satisfy dependency"""
    if goal is None:
        lv = 1
        while lv in goal_info['lv_tasks']:
            goals = goal_info['lv_tasks'][lv]
            for g in goals:
                if g not in goal_info['pred']:
                    continue
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
                if g not in goal_info['succ']:
                    continue
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

def extend_paths(paths, window, needs_replan):
    """Extend paths to a fixed length by appending stay actions"""
    for i, path in enumerate(paths):
        if needs_replan[i]:
            pos = (path[-1][0], 'move')
            path += [pos] * (window - len(path))


'''Conflict handling'''
def detect_all_conflicts(info, heights, paths, block_actions, detect_mode, priority):
    """Detect conflicts between all pairs of paths"""
    heights = construct_heights(heights, block_actions, info)
    conflicts = []
    for i in range(len(paths)):
        if not info['needs_replan'][i]:
            continue
        for j in range(i + 1, len(paths)):
            if not info['needs_replan'][j]:
                continue
            c = detect_conflict(heights, paths[i], paths[j], block_actions[i], block_actions[j], detect_mode)
            if c:
                c['a1'] = i
                c['a2'] = j
                c['priority'] = priority[c['type']]
                conflicts.append(c)
        c = detect_height_conflict(heights, paths[i], block_actions, i, info['start'])
        if c:
            c['a1'] = i
            c['priority'] = priority[c['type']]
            conflicts.append(c)
    return conflicts

def detect_conflict(heights, path1, path2, block_action1, block_action2, mode):
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
            height = heights[t] if t < heights.shape[0] else heights[-1]
            loc1, loc2 = path1[t][0][:2], path2[t][0][:2]

            if r == 0:
                '''Edge-block conflict: a1 moves from A to B, while a2 performs block action from B to A'''
                if t == t1 and gloc1 == ploc2 and loc1 == loc2:
                    return {'type': 'edge-block', 'time': t, 'loc': (loc1, gloc1), 'block': 1, 'rt': t-1}
                if t == t2 and gloc2 == ploc1 and loc1 == loc2:
                    return {'type': 'edge-block', 'time': t, 'loc': (loc2, gloc2), 'block': 2, 'rt': t-1}

                '''Agent-block conflict: a1 moves from A to B, while a2 performs block action at B'''
                if t == t1:
                    if gloc1 == ploc2 == loc2:  # Agent 2 staying at agent 1's block location
                        return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'stay', 'rt': t-1}
                    if gloc1 == ploc2:  # Agent 2 leaving agent 1's block location
                        return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'leave', 'rt': t-1}
                    if gloc1 == loc2:  # Agent 2 arriving at agent 1's block location
                        return {'type': 'agent-block', 'time': t, 'loc': gloc1, 'block': 1, 'move': 'arrive', 'rt': t}
                if t == t2:
                    if gloc2 == ploc1 == loc1:
                        return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'stay', 'rt': t-1}
                    if gloc2 == ploc1:
                        return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'leave', 'rt': t-1}
                    if gloc2 == loc1:
                        return {'type': 'agent-block', 'time': t, 'loc': gloc2, 'block': 2, 'move': 'arrive', 'rt': t}

                '''Move height conflict: a1 moves from A to B, B is a2's goal location, with height difference > 1'''
                h1, h2 = height[loc1], height[loc2]
                ph1, ph2 = height[ploc1], height[ploc2]
                # TODO: do we need this?
                # if gloc1 == ploc2 and ((ph2 < h2 - 1 and lv1 == ph2 + 1) or (ph2 > h2 + 1 and lv1 == ph2 - 2)):
                #     return {'type': 'move-height', 'time': t-1, 'loc': (ploc2, loc2), 'block': 1, 'tb': t1, 'rt': min(t-1, t1)}
                # if gloc2 == ploc1 and ((ph1 < h1 - 1 and lv2 == ph1 + 1) or (ph1 > h1 + 1 and lv2 == ph1 - 2)):
                #     return {'type': 'move-height', 'time': t-1, 'loc': (ploc1, loc1), 'block': 2, 'tb': t2, 'rt': min(t-1, t2)}
                if gloc1 == loc2 and ((ph2 < h2 - 1 and lv1 == ph2 + 1) or (ph2 > h2 + 1 and lv1 == ph2 - 2)):
                    return {'type': 'move-height', 'time': t, 'loc': (ploc2, loc2), 'block': 1, 'tb': t1, 'rt': min(t, t1)}
                if gloc2 == loc1 and ((ph1 < h1 - 1 and lv2 == ph1 + 1) or (ph1 > h1 + 1 and lv2 == ph1 - 2)):
                    return {'type': 'move-height', 'time': t, 'loc': (ploc1, loc1), 'block': 2, 'tb': t2, 'rt': min(t, t2)}

                '''Block height conflict: a1 performs block action from A to B, A = g2, while height is incorrect'''
                # Assumption: block height at B is correct
                if t == t1 and loc1 == gloc2 and lv1 != h1 and (1 >= lv1 - lv2 >= 0):
                    return {'type': 'block-height', 'time': t, 'loc': loc1, 'curr': 1, 't2': t2, 'rt': min(t1, t2)}
                if t == t2 and loc2 == gloc1 and lv2 != h2 and (1 >= lv2 - lv1 >= 0):
                    return {'type': 'block-height', 'time': t, 'loc': loc2, 'curr': 2, 't2': t1, 'rt': min(t1, t2)}

            if (r == 0 and mode == 0) or (r == 1 and mode == 1):
                '''No conflict outside the world'''
                if ploc1[0] == -1 or ploc2[0] == -1 or loc1[0] == -1 or loc2[0] == -1:
                    pass
                else:
                    '''Vertex conflict'''
                    if loc1 == loc2 and loc1[0] != -1:
                        return {'type': 'vertex', 'time': t, 'loc': loc1, 'rt': t}
                    '''Edge conflict'''
                    if loc1 == ploc2 and loc2 == ploc1 and loc1[0]:
                        return {'type': 'edge', 'time': t, 'loc': (ploc1, loc1), 'rt': t}
                    '''Block-block conflict: ignored, won't happen'''

            ploc1, ploc2 = loc1, loc2
    return None

def detect_height_conflict(heights, path, block_actions, i, start):
    """Detect height conflicts with fixed paths"""
    tb, (add, gx, gy, lv, _) = block_actions[i]
    ploc = path[start][0][:2]
    for t in range(start + 1, len(path)):
        height = heights[t] if t < heights.shape[0] else heights[-1]
        loc = path[t][0][:2]
        h, ph = height[loc], height[ploc]
        if abs(h - ph) > 1:
            fixed_con = True
            for t2, (add2, x2, y2, lv2, _) in block_actions:
                if gx == x2 and gy == y2:
                    continue
                if (x2, y2) != loc:
                    continue
                if (ph < h - 1 and lv2 == ph + 1) or (ph > h + 1 and lv2 == ph - 2):
                    fixed_con = False
                    break
            if fixed_con:
                return {'type': 'move-height-fixed', 'time': t, 'loc': (ploc, loc), 'rt': t}
        if t == tb and h != lv:
            fixed_con = True
            for t2, (add2, x2, y2, lv2, _) in block_actions:
                if gx == x2 and gy == y2:
                    continue
                if (x2, y2) != loc:
                    continue
                if 1 >= lv2 - lv >= 0:
                    fixed_con = False
                    break
            if fixed_con:
                return {'type': 'block-height-fixed', 'time': t, 'loc': loc, 'rt': t}
        ploc = loc
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
        tb = conflict['tb']
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
    Mode 1: cost = makespan = max of time steps until goal, (early rounds), max of time steps until parking (last round)
    Mode 2: cost = makespan = min of time steps until goal, (early rounds), max of time steps until parking (last round)
    Mode 3: cost = (mode 2, mode 1)
    """
    if mode == 0:
        cost = 0
        for t, _ in block_actions:
            cost += t
    elif mode == 1:
        if last_round:
            cost = len(paths[0])
        else:
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
    elif mode == 3:
        cost1 = compute_cost(paths, block_actions, 2, last_round)
        cost2 = compute_cost(paths, block_actions, 1, last_round)
        cost = (cost1, cost2)
    else:
        raise NotImplementedError
    return cost

def order_conflicts(conflicts, mode):
    """
    Order conflicts between paths
    Mode 0: default order, use agent index
    Mode 1: conflict time order, solve conflicts happening earlier first
    Mode 2: constraint time order, solve conflicts that produce earlier constraints first
    Mode 3: constraint type order, (block, height) -> (vertex, edge)
    Mode 4: constraint type order, (block) -> (height) -> (vertex, edge)
    Mode 5: constraint type order, (height) -> (block) -> (vertex, edge)
    """
    if mode == 0:
        pass
    elif mode == 1:
        conflicts.sort(key=lambda x: x['time'])
    elif mode == 2:
        conflicts.sort(key=lambda x: x['rt'])
    elif mode >= 3:
        conflicts.sort(key=lambda x: x['priority'])
    else:
        raise NotImplementedError

def push_node(open_list, node):
    """Push a node into the open list. Order = cost, # conflicts, gen_id"""
    heapq.heappush(open_list, (node.cost, len(node.conflicts), node.gen_id, node))

def cbs(env, info, arg, last_round, heights=None):
    assert arg.detect <= 1 and arg.resolve <= 5
    # height = env.height
    num = len(info['goals'])
    limit = env.w * env.w

    '''Plan initial single-agent paths'''
    paths, times = [], []
    for i in range(num):
        if info['needs_replan'][i]:
            path, t = a_star(env, info, heights, [], i, arg, earliest=info['earliest'])
            path = info['planned_paths'][i] + path[1:]
            paths.append(path)
            times.append(t + info['available_t'][i])
        else:
            paths.append([])
            times.append(0)
    paths, times = insert_stays(info, paths, times)
    window = max([len(p) for p in paths])
    extend_paths(paths, window, info['needs_replan'])
    block_actions = [(times[i], info['goals'][i]) for i in range(num)]
    root = Node(paths, times, block_actions, set(), 0)
    root.cost = compute_cost(paths, block_actions, arg.cost, last_round)
    root.conflicts = detect_all_conflicts(info, heights, paths, block_actions, arg.detect, arg.priority)

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
            goal = info['goals'][aid]
            if goal in info['pred']:
                pred = info['pred'][goal]
                earliest = 0 if len(pred) == 0 else max(node.times[info['id'][p]] for p in pred) + 1
            else:
                earliest = 0
            path, t = a_star(env, info, heights, child.constraints, aid, arg, earliest=earliest, latest=limit, paths=child.paths, block_actions=child.block_actions)
            if path:
                child.paths[aid] = info['planned_paths'][aid] + path[1:]
                child.times[aid] = t
                child.paths, child.times = insert_stays(info, child.paths, child.times, info['goals'][aid])
                window = max([len(p) for p in child.paths])
                extend_paths(child.paths, window, info['needs_replan'])
                child.block_actions = [(child.times[i], info['goals'][i]) for i in range(num)]
                child.cost = compute_cost(child.paths, child.block_actions, arg.cost, last_round)
                child.conflicts = detect_all_conflicts(info, heights, child.paths, child.block_actions, arg.detect, arg.priority)

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
