import pickle as pk

import lego
import config
from high_plan import high_lv_plan
from cbs import cbs


def task_allocation(todo, assignment):
    # Select candidate tasks
    tasks = todo[:min(arg.num, len(todo))]
    tasks, num_dummy = adjust_task(tasks)
    # Allocate tasks to agents
    i = 0
    for a in range(arg.num):
        if assignment[a] is None:
            while tasks[i] in assignment:
                i += 1
            assignment[a] = tasks[i]
    return assignment, num_dummy

def adjust_task(tasks):
    # Delay a task if it deals with the same block as another task
    delay = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i][1:] == tasks[j][1:]:
                delay.append(j)
    final_tasks = []
    for i in range(len(tasks)):
        if i not in delay:
            final_tasks.append(tasks[i])
    # Append dummy tasks if there are fewer tasks than agents
    num_dummy = arg.num - len(final_tasks)
    final_tasks += num_dummy * [(-1, -1, -1, -1)]
    return final_tasks, num_dummy

def add_carry(path, carry):
    for i in range(len(path)):
        if path[i][2] in ['depo', 'goal']:
            carry = not carry
        path[i] = (path[i][0], path[i][1], path[i][2], carry)
    return path, carry

def plan():
    finish, total = 0, len(high_path)
    todo = high_path.copy()
    goals = [None for _ in range(arg.num)]
    full_paths = [[] for _ in range(arg.num)]
    while finish < total:
        goals, num_dummy = task_allocation(todo, goals)
        paths, times = cbs(env, goals, positions, carry_stats)
        '''Determine execution length'''
        if arg.num - num_dummy < len(todo):  # There are tasks not assigned
            execute_t = len(paths[0])  # Execute until the assigned first task is finished
            for t in times:
                if t > 0:
                    execute_t = min(execute_t, t)
            for i in range(arg.num):
                executed = paths[i][:execute_t + 1]
                executed, carry = add_carry(executed, carry_stats[i])
                full_paths[i] += executed
                positions[i] = executed[-1][:2]
                carry_stats[i] = carry
                if times[i] == execute_t:
                    finish += 1
                    add, x, y, lv = goals[i]
                    if add:
                        env.height[x, y] += 1
                    else:
                        env.height[x, y] -= 1
                    todo.remove(goals[i])
                    goals[i] = None
        else:  # All tasks are assigned
            for i in range(arg.num):
                path, carry = add_carry(paths[i], carry_stats[i])
                full_paths[i] += path
                add, x, y, lv = goals[i]
                if add == 0:
                    env.height[x, y] -= 1
                elif add == 1:
                    env.height[x, y] += 1
            finish = total
    return full_paths


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    positions = [(0, 1), (0, 2)]
    carry_stats = [True, True]
    # positions = [(0, 1), (0, 2), (0, 3)]
    # carry_stats = [True, True, True]

    if arg.start == 0:
        high_path = high_lv_plan(env)
    else:
        with open('high_plan.pkl', 'rb') as f:
            high_path = pk.load(f)
    print(high_path)

    paths = plan()

    # for p in paths:
    #     print(p)
