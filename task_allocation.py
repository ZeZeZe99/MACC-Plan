import numpy as np
from scipy.optimize import linear_sum_assignment

from path_finding_2d import distance2border, distance2neighbor, movable_neighbor, heuristic

"""
Task selection methods
Mode 0: Sequence selection
    1. Sequentially select tasks from the original action sequence
    2. Add naive precedence constraint to the previous task at the same x-y coordinate
Mode 1: Level selection (k)
    1. Take the reordered action sequence, which breaks into several levels
    2. Sequentially select tasks from each level, up to k levels
    3. Enforce higher levels to be executed after lower levels
"""


'''Select new tasks'''
def select_tasks(num, assigned, todo, mode=0, k=1):
    if mode == 0:
        new_tasks, priority = sequence_selection(num, assigned, todo)
    elif mode == 1:
        new_tasks, priority = level_selection(num, assigned, todo, k=k)
    else:
        new_tasks, priority = dependency_selection(num, todo, assigned, level=k)
    return new_tasks, priority

def sequence_selection(num, assigned, seq):
    tasks = seq[:min(num, len(seq))]
    new_tasks = [t for t in tasks if t not in assigned]

    '''Add naive priority (precedence constraint between tasks at the same x-y coordinate)'''
    lv_task, task_lv, predecessors, successors, all_succ = {}, {}, {}, {}, {}
    for i in range(len(tasks)):
        lv = 0
        predecessors[tasks[i]] = set()
        successors[tasks[i]] = set()
        xi, yi = tasks[i][1:3]
        for j in range(i - 1, -1, -1):
            xj, yj = tasks[j][1:3]
            if xj == xi and yj == yi:
                lv = task_lv[tasks[j]] + 1
                predecessors[tasks[i]].add(tasks[j])
                successors[tasks[j]].add(tasks[i])
                break
        ''''''
        if lv not in lv_task:
            lv_task[lv] = set()
        lv_task[lv].add(tasks[i])
        task_lv[tasks[i]] = lv
    for i in range(len(tasks) - 1, -1, -1):
        all_succ[tasks[i]] = successors[tasks[i]].copy()
        for j in successors[tasks[i]]:
            all_succ[tasks[i]] |= all_succ[j]
    info = {'lv_task': lv_task, 'task_lv': task_lv, 'pred': predecessors, 'succ': successors,
            'all_succ': all_succ}
    return new_tasks, info

def level_selection(num, assigned, level, k=1):
    assert k >= 0
    tasks = []
    lv_tasks, task_lv, predecessors, successors, all_succ = {}, {}, {}, {}, {}
    i = 0
    for lv in range(max(level.keys()) + 1):
        if len(level[lv]) == 0:
            continue
        if i == k or len(tasks) == num:
            break
        lv_tasks[i] = level[lv][:min(num - len(tasks), len(level[lv]))]
        tasks += lv_tasks[i]
        for t in lv_tasks[i]:
            task_lv[t] = i
        i += 1
    for lv in range(i - 1, -1, -1):
        for t in lv_tasks[lv]:
            if lv + 1 in lv_tasks:
                successors[t] = set(lv_tasks[lv + 1])
                all_succ[t] = set()
                for s in lv_tasks[lv + 1]:
                    all_succ[t] |= all_succ[s]
            else:
                successors[t] = set()
                all_succ[t] = set()
            if lv - 1 in lv_tasks:
                predecessors[t] = set(lv_tasks[lv - 1])
            else:
                predecessors[t] = set()
    predecessors[(-1, -1, -1, -1, -1)] = set()
    successors[(-1, -1, -1, -1, -1)] = set()
    all_succ[(-1, -1, -1, -1, -1)] = set()
    info = {'lv_tasks': lv_tasks, 'task_lv': task_lv, 'pred': predecessors, 'succ': successors, 'all_succ': all_succ}
    new_tasks = [t for t in tasks if t not in assigned]
    return new_tasks, info

def dependency_selection(num, g, assigned, level=1):
    assert level >= 0
    tasks = assigned.copy()  # Assigned tasks

    '''Find candidate tasks'''
    temp_g = g.copy()
    leaf_node = []
    dep_lv_2_g, g_2_dep_lv, predecessors, successors, all_succ = {}, {}, {}, {}, {}
    lv = 0
    stop = 999 if level == 0 else level
    while len(tasks) < num:
        if lv == stop:
            break
        temp_g.remove_nodes_from(leaf_node)
        leaf_node = [n for n in temp_g.nodes if temp_g.in_degree[n] == 0]  # Current leaf nodes
        leaf_assigned = [n for n in leaf_node if n in tasks]
        leaf_new = [n for n in leaf_node if n not in tasks]
        '''If tasks at this level are more than required, select some of them'''
        leaf_new = leaf_new[:min(len(leaf_new), num - len(tasks))]
        leaf_selected = leaf_assigned + leaf_new
        dep_lv_2_g[lv] = set(leaf_selected)
        for n in leaf_selected:
            g_2_dep_lv[n] = lv
        lv += 1
        tasks += leaf_new

    '''Record predecessors and successors of each task'''
    max_lv = lv - 1
    for lv in range(max_lv, -1, -1):
        for n in dep_lv_2_g[lv]:
            successors[n] = set([s for s in g.successors(n) if s in tasks])
            all_succ[n] = set()
            for s in successors[n]:
                all_succ[n] |= all_succ[s] if s in all_succ else set()
            all_succ[n] |= successors[n]
            predecessors[n] = set([p for p in g.predecessors(n) if p in tasks])
    predecessors[(-1, -1, -1, -1, -1)] = set()
    successors[(-1, -1, -1, -1, -1)] = set()
    all_succ[(-1, -1, -1, -1, -1)] = set()

    new_tasks = [t for t in tasks if t not in assigned]
    info = {'lv_tasks': dep_lv_2_g, 'task_lv': g_2_dep_lv, 'pred': predecessors, 'succ': successors,
           'all_succ': all_succ}
    return new_tasks, info


'''Process task information'''
def preprocess_tasks(tasks, info, env):
    """Pre-process task information, assuming tasks are fixed"""
    '''Mark possible heights for each x-y location'''
    loc2height = dict()
    for loc in env.valid_loc:
        loc2height[loc] = {env.height[loc]}
    for gadd, gx, gy, lv, _ in tasks:
        if gadd == 1:
            loc2height[gx, gy].add(lv + 1)
        elif gadd == 0:
            loc2height[gx, gy].add(lv)
    '''Mark possible neighbors for each goal (where the agent can stand at while performing the goal action)'''
    g2neighbor = dict()
    for _, gx, gy, lv, _ in tasks:
        if gx != -1:
            g2neighbor[(gx, gy, lv)] = set()
            for (nx, ny) in env.valid_neighbor[gx, gy]:
                nzs = loc2height[(nx, ny)]
                if lv in nzs:
                    g2neighbor[(gx, gy, lv)].add((nx, ny))
            assert len(g2neighbor[(gx, gy, lv)]) > 0
    '''Pre-compute distance heuristic'''
    info['d2border'] = distance2border(env, loc2height)
    info['d2neighbor'] = distance2neighbor(env, loc2height, g2neighbor)
    info['loc_height'] = loc2height
    info['goal_work_loc'] = g2neighbor
    info['movable'] = movable_neighbor(env, loc2height)

def postprocess_tasks(info):
    """Post-process task information, assuming tasks are assigned"""
    tasks = info['goals']
    '''Task index'''
    info['id'] = dict()
    for i in range(len(tasks)):
        info['id'][tasks[i]] = i
    '''Mark start stage for each agent'''
    info['stage'] = dict()
    for i in range(len(tasks)):
        add = tasks[i][0]
        carry = info['carry'][i]
        if add < 0:
            info['stage'][i] = 2  # No goal, go to border
        elif add == carry:
            info['stage'][i] = 1  # Can directly perform goal action
        else:
            info['stage'][i] = 0  # Need to go to border first


'''Allocate tasks to agents'''
def allocate_tasks(assignment, tasks, info, env, arg):
    old_tasks = [t for t in assignment if t is not None]
    preprocess_tasks(old_tasks + tasks, info, env)
    if arg.allocate == 0:
        assignment = naive_allocate(assignment, tasks, arg)
    elif arg.allocate == 1:
        assignment = matching_allocate(assignment, tasks, info, env, arg)
    else:
        raise NotImplementedError
    info['goals'] = assignment
    postprocess_tasks(info)
    return assignment

def naive_allocate(assignment, tasks, arg):
    """Naively assign tasks to agents in order"""
    if arg.reselect:
        assignment = [None for _ in range(len(assignment))]
    ids = [i for i in range(len(assignment)) if assignment[i] is None]
    tid = 0
    for i in range(len(ids)):
        if tid < len(tasks):
            assignment[ids[i]] = tasks[tid]
            tid += 1
        else:
            aid = ids[i]
            assignment[aid] = (-1, -1, -1, -1, -1)
    return assignment

def matching_allocate(assignment, tasks, info, env, arg):
    """Match tasks to agents based on cost estimation"""
    prev = assignment.copy()
    if arg.reselect:
        assignment = [None] * len(assignment)
    ids = [i for i in range(len(assignment)) if assignment[i] is None]
    if len(ids) > len(tasks):
        tasks.append((-1, -1, -1, -1, -1))
    costs = np.zeros((len(ids), len(tasks)), dtype=np.float16)
    for aid in range(len(ids)):
        for tid in range(len(tasks)):
            costs[aid, tid] = estimate_cost(info, env, tasks[tid], aid, arg)
            if prev[aid] is not None and prev[aid] == tasks[tid]:
                costs[aid, tid] -= 0.5
    if len(ids) > len(tasks):
        dummy_cost = np.tile(costs[:, -1:], (1, len(ids) - len(tasks)))
        costs = np.concatenate((costs, dummy_cost), axis=1)
        tasks += [(-1, -1, -1, -1, -1)] * (len(ids) - len(tasks))
    row, col = linear_sum_assignment(costs)
    for i in range(len(row)):
        assignment[ids[row[i]]] = tasks[col[i]]
    return assignment

def estimate_cost(info, env, task, aid, arg):
    x, y = info['pos'][aid]
    carry = info['carry'][aid]
    add, gx, gy, lv, _ = task
    if add < 0:
        stage = 2
    elif add == carry:
        stage = 1
    else:
        stage = 0
    cost = heuristic(env, info, stage, x, y, gx, gy, lv, arg.heu, arg.teleport)[1]
    return cost


'''Remove completed tasks'''
def remove_tasks(tasks, t, mode):
    if mode == 0:
        tasks.remove(t)
    elif mode == 1:
        for lv in tasks:
            if t in tasks[lv]:
                tasks[lv].remove(t)
                break
    else:
        tasks.remove_node(t)
