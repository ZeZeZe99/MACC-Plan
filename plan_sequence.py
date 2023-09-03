from task_allocation import select_tasks, allocate_tasks, remove_tasks
from cbs_2d import cbs

def plan(env, arg, tasks):
    num = arg.num
    positions = [(-1, -1) for _ in range(num)]
    carry_stats = [True for _ in range(num)]
    if arg.select == 0:
        total = len(tasks)
    elif arg.select == 1:
        total = 0
        for lv in tasks:
            total += len(tasks[lv])
    else:
        total = len(tasks.nodes)
    assignment = [None for _ in range(num)]
    full_paths = [[(positions[i], carry_stats[i], None)] for i in range(num)]
    finish = generate = expand = 0

    while finish < total:
        assigned = [] if arg.reselect else [t for t in assignment if t is not None]
        new_tasks, info = select_tasks(num, assigned, tasks, mode=arg.select, k=arg.k)
        last_round = len(assigned) + len(new_tasks) == total - finish

        info['pos'] = positions
        info['carry'] = carry_stats
        assignment = allocate_tasks(assignment, new_tasks, info, env, arg)
        paths, times, stat = cbs(env, info, arg, last_round)
        generate += stat[0]
        expand += stat[1]
        '''Determine execution length'''
        if not last_round:  # There are tasks not assigned
            execute_t = execute_time(times, info, arg.execute)
            for i in range(num):
                executed = paths[i][1:execute_t + 1]
                executed, carry = snapshot(executed, carry_stats[i], assignment[i])
                full_paths[i] += executed
                positions[i] = executed[-1][0]
                carry_stats[i] = carry
                if times[i] == -1:
                    assignment[i] = None
                elif times[i] <= execute_t:
                    finish += 1
                    add, x, y, lv, _ = assignment[i]
                    if add:
                        env.height[x, y] += 1
                    else:
                        env.height[x, y] -= 1
                    remove_tasks(tasks, assignment[i], mode=arg.select)
                    assignment[i] = None
        else:  # All tasks are assigned
            for i in range(num):
                path, carry = snapshot(paths[i][1:], carry_stats[i], assignment[i])
                full_paths[i] += path
                add, x, y, lv, _ = assignment[i]
                if add == 0:
                    env.height[x, y] -= 1
                elif add == 1:
                    env.height[x, y] += 1
            finish = total
    return full_paths, (generate, expand)

def execute_time(times, info, mode):
    if mode == 0:
        return max(times)
    elif mode == 1:
        execute_t = max(times)
        for t in times:
            if t > 0:
                execute_t = min(execute_t, t)
    else:
        execute_t = max(times)
        for task in info['lv_tasks'][0]:
            t = times[info['goals'].index(task)]
            if t > 0:
                execute_t = min(execute_t, t)
    return execute_t

def snapshot(path, carry, goal):
    """Convert (loc, action) to (loc, carry, goal)"""
    for i in range(len(path)):
        if path[i][1] in ['depo', 'goal']:
            carry = not carry
        if path[i][1] == 'goal':
            g = goal[:4]
        else:
            g = None
        path[i] = (path[i][0], carry, g)
    return path, carry