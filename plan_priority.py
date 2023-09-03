import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from path_finding_3d import distance2border, distance2neighbor, available_action, heuristic
from cbs_3d import cbs

def priority_plan(env, lv_tasks, arg):
    num = arg.num
    result_paths = [[((-1, -1), True, None)] for _ in range(num)]
    paths = [[((-1, -1, -1), 'move')] for _ in range(num)]
    extend_paths = [[((-1, -1, -1), 'move')] for _ in range(num)]

    level = last_done = 0
    available_t = np.zeros(num, dtype=np.int8)
    available_pos = [(-1, -1, -1) for _ in range(num)]
    available_carry = np.ones(num, dtype=bool)
    done = dict()
    heights = np.zeros((1, *env.world_shape), dtype=np.int8)
    fixed_cons = dict()
    while level < len(lv_tasks):
        info = dict()
        info['available_t'] = available_t
        info['pos'] = available_pos
        info['carry'] = available_carry
        info['planned_paths'] = paths

        '''Task selection: first N tasks at current level'''
        num_tasks = min(num, len(lv_tasks[level]))
        curr_tasks = lv_tasks[level][:num_tasks]
        lv_tasks[level] = lv_tasks[level][num_tasks:]
        info['lv_tasks'] = {0: curr_tasks}
        info['pred'], info['succ'], info['all_succ'] = dict(), dict(), dict()

        '''Task allocation'''
        start_est = np.min(available_t)  # Start time for cost estimation
        info['earliest'] = last_done + 1  # Earliest new task execution time
        process_workspace(env, info, curr_tasks, start_est, heights, done)
        assignment = matching(env, curr_tasks, info, arg)
        needs_replan = np.array([task[0] != -1 for task in assignment], dtype=bool)

        '''Task postprocessing'''
        start = np.min(available_t + 1000 * (1 - needs_replan))  # Start time for path finding
        if start != start_est:
            process_workspace(env, info, curr_tasks, start, heights, done)
        info['start'] = start
        info['needs_replan'] = needs_replan
        postprocess(info, assignment)
        valid_action(env, info, heights)
        prior_cons = deepcopy(fixed_cons)
        for i in range(num):
            if not needs_replan[i]:
                add_constraints(prior_cons, extend_paths[i], available_t[i])
        info['prior_cons'] = prior_cons
        # if level == 4:
        #     info['available_t'] = 6 * np.ones(num, dtype=np.int8)
        #     info['planned_paths'] = [[((-1, -1, -1), 'move') for _ in range(7)] for _ in range(num)]
        #     info['prior_cons'] = dict()
        #     info['pos'] = [(-1, -1, -1) for _ in range(num)]

        '''Path finding'''
        last_round = (level == len(lv_tasks) - 1) and len(lv_tasks[level]) == 0
        new_paths, times, _ = cbs(env, info, arg, last_round, heights)

        '''Update results'''
        if max(times) > heights.shape[0] - 1:  # Pad height map
            new_heights = np.tile(heights[-1], (max(times) - heights.shape[0] + 1, 1, 1))
            heights = np.concatenate((heights, new_heights), axis=0)
        for i in range(num):
            if needs_replan[i]:
                add, x, y, lv, _ = assignment[i]
                heights[times[i]:, x, y] += 1 if add == 1 else -1  # Update height map
                extend_paths[i] = paths[i] + new_paths[i][available_t[i] + 1:]  # Update extended paths
                path = new_paths[i][available_t[i] + 1:times[i] + 1]
                paths[i] += path  # Update paths
                result_paths[i] += snapshot(path, assignment[i], available_carry[i])  # Update result paths
                available_pos[i] = path[-1][0]  # Update position
                available_carry[i] = not add  # Update carry status
                add_constraints(fixed_cons, paths[i], available_t[i], times[i], (x, y))  # Update constraints
                available_t[i] = times[i]  # Update available time
                if times[i] not in done:
                    done[times[i]] = set()
                done[times[i]].add(assignment[i])  # Update done tasks

        if len(lv_tasks[level]) == 0:
            level += 1
            last_done = max(available_t)
    '''Pad result paths'''
    for i in range(num):
        path = extend_paths[i][available_t[i] + 1:]
        result_paths[i] += snapshot(path, (-1, -1, -1, -1, -1), available_carry[i])
    max_t = max([len(path) for path in result_paths])
    for i in range(num):
        result_paths[i] += [((-1, -1), available_carry[i], None)] * (max_t - len(result_paths[i]))
    return result_paths, None


def add_constraints(cons, path, start_t, goal_t=None, goal_loc=None):
    """Add priority constraints"""
    '''Vertex constraints'''
    for t in range(start_t + 1, len(path)):
        if t not in cons:
            cons[t] = [set(), set()]  # Set 0: agent + block locations; Set 1: agent locations
        x, y, _ = path[t][0]
        if x == -1:
            continue
        cons[t][0].add((x, y))
        cons[t][1].add((x, y))
        if goal_t and goal_t - t <= 1:
            cons[t][0].add(goal_loc)
    '''Edge constraints'''
    for t in range(start_t + 1, len(path)):
        x1, y1, _ = path[t - 1][0]
        x2, y2, _ = path[t][0]
        if x1 == -1 or x2 == -1:
            continue
        cons[t][0].add(((x2, y2), (x1, y1)))


'''Process task information'''
def process_workspace(env, info, tasks, start, heights, done):
    """Process workspace information (heights, distances, etc.)"""
    '''Mark possible heights for each x-y location'''
    all_loc_height = dict()  # Include both done and new tasks
    new_loc_height = dict()  # Only include new tasks
    for loc in env.valid_loc:  # Use height at t = start as the initial height
        all_loc_height[loc] = {heights[start, loc[0], loc[1]]}
    for t in done:  # Consider done tasks at t >= start
        if t > start:
            done_tasks = done[t]
            for add, x, y, lv, _ in done_tasks:
                all_loc_height[x, y].add(lv + add)
    for add, x, y, lv, _ in tasks:  # Consider new tasks
        if x != -1:
            all_loc_height[x, y].add(lv + add)
            new_loc_height[x, y] = lv + add
    '''Mark possible work locations for each goal (where the agent can stand at while performing the goal action)'''
    goal_work_loc = dict()
    for _, x, y, lv, _ in tasks:
        if x != -1:
            goal_work_loc[x, y, lv] = set()
            for nx, ny in env.valid_neighbor[x, y]:
                if lv in all_loc_height[nx, ny]:
                    # goal_work_loc[x, y, lv].add((nx, ny))
                    goal_work_loc[x, y, lv].add((nx, ny, lv))
            assert len(goal_work_loc[x, y, lv]) > 0
    '''Pre-compute distance heuristic'''
    info['d2border'] = distance2border(env, all_loc_height)
    info['d2neighbor'] = distance2neighbor(env, all_loc_height, goal_work_loc)
    info['task_height'] = new_loc_height
    info['goal_work_loc'] = goal_work_loc

def postprocess(info, assignment):
    info['goals'] = assignment
    '''Task index'''
    info['id'] = dict()
    for i in range(len(assignment)):
        info['id'][assignment[i]] = i
    '''Mark start stage for each agent'''
    info['stage'] = dict()
    for i in range(len(assignment)):
        add = assignment[i][0]
        carry = info['carry'][i]
        if add < 0:
            info['stage'][i] = 2  # No goal, go to border
        elif add == carry:
            info['stage'][i] = 1  # Can directly perform goal action
        else:
            info['stage'][i] = 0  # Need to go to border first

def valid_action(env, info, heights):
    """Mark available actions at each time step"""
    actions = dict()
    for t in range(info['start'] + 1, heights.shape[0] + 1):
        if t > info['earliest'] or t == heights.shape[0]:
            if t - 1 > info['earliest'] and np.array_equal(heights[t], heights[t - 1]):
                actions[t] = actions[t - 1]
                continue
            actions[t] = dict()
            height = heights[t] if t < heights.shape[0] else heights[-1]
            for x, y in env.valid_loc:
                z = z2 = height[x, y]
                actions[t][x, y, z] = []
                if (x, y) in info['task_height']:
                    z2 = info['task_height'][x, y]
                    actions[t][x, y, z2] = []
                for nx, ny in env.valid_neighbor[x, y]:
                    if abs(z - height[nx, ny]) <= 1:
                        actions[t][x, y, z].append((nx, ny, height[nx, ny]))
                    if z2 != z and abs(z2 - height[nx, ny]) <= 1:
                        actions[t][x, y, z2].append((nx, ny, height[nx, ny]))
                    if (nx, ny) in info['task_height']:
                        nz = info['task_height'][nx, ny]
                        if abs(z - nz) <= 1:
                            actions[t][x, y, z].append((nx, ny, nz))
                        if z2 != z and abs(z2 - nz) <= 1:
                            actions[t][x, y, z2].append((nx, ny, nz))
            actions[t][-1, -1, -1] = []
        else:
            if t - 1 in actions.keys() and np.array_equal(heights[t], heights[t - 1]):
                actions[t] = actions[t - 1]
                continue
            actions[t] = dict()
            height = heights[t]
            for x, y in env.valid_loc:
                z = height[x, y]
                actions[t][x, y, z] = []
                for nx, ny in env.valid_neighbor[x, y]:
                    if abs(z - height[nx, ny]) <= 1:
                        actions[t][x, y, z].append((nx, ny, height[nx, ny]))
            actions[t][-1, -1, -1] = []
    info['actions'] = actions


'''Task assignment'''
def matching(env, tasks, info, arg):
    """Match tasks to agents based on estimation of completion time"""
    if len(tasks) < arg.num:
        tasks.append((-1, -1, -1, -1, -1))
    estimate_t = np.zeros((arg.num, len(tasks)), dtype=np.int8)
    for i in range(arg.num):
        for j in range(len(tasks)):
            cost = estimate_cost(env, info, tasks[j], info['pos'][i], info['carry'][i], arg)
            if tasks[j][0] == -1:
                estimate_t[i, j] = 0
            else:
                estimate_t[i, j] = max(info['available_t'][i] + cost, info['earliest'])
    if len(tasks) < arg.num:
        dummy = np.tile(estimate_t[:, -1:], (1, arg.num - len(tasks)))
        estimate_t = np.concatenate((estimate_t, dummy), axis=1)
        tasks += [(-1, -1, -1, -1, -1)] * (arg.num - len(tasks))
    '''Hungarian algorithm'''
    row, col = linear_sum_assignment(estimate_t)
    assignment = [tasks[col[i]] for i in range(arg.num)]
    return assignment

def estimate_cost(env, info, task, pos, carry, arg):
    x, y, z = pos
    add, gx, gy, lv, _ = task
    if add < 0:
        stage = 2
    elif add == carry:
        stage = 1
    else:
        stage = 0
    cost = heuristic(env, info, stage, x, y, z, gx, gy, lv, arg.heu, arg.teleport)[1]
    return cost


'''Result'''
def snapshot(path, task, carry):
    new_path = []
    for i in range(len(path)):
        if path[i][1] in ['depo', 'goal']:
            carry = not carry
        if path[i][1] == 'goal':
            g = task[:4]
        else:
            g = None
        new_path.append((path[i][0][:2], carry, g))
    return new_path
