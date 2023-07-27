import pickle as pk
import cProfile
import pstats

import lego
import config
from high_plan_astar import high_lv_plan
from cbs import cbs
from task_allocation import allocate

def snapshot(path, carry, goal):
    """Convert (loc, action) to (loc, carry, goal)"""
    for i in range(len(path)):
        if path[i][1] in ['depo', 'goal']:
            carry = not carry
        if path[i][1] == 'goal':
            g = goal
        else:
            g = None
        path[i] = (path[i][0], carry, g)
    return path, carry

def plan():
    finish, total = 0, len(high_path)
    todo = high_path.copy()
    goals = [None for _ in range(num)]
    full_paths = [[(positions[i], carry_stats[i], None)] for i in range(num)]
    flow_time = generate = expand = 0
    while finish < total:
        goals, num_dummy = allocate(num, todo, goals)
        paths, times, stat = cbs(env, goals, positions, carry_stats)
        generate += stat[0]
        expand += stat[1]
        '''Determine execution length'''
        if num - num_dummy < len(todo):  # There are tasks not assigned
            execute_t = len(paths[0])  # Execute until the assigned first task is finished
            for t in times:
                if t > 0:
                    execute_t = min(execute_t, t)
            flow_time += execute_t * num
            for i in range(num):
                executed = paths[i][1:execute_t + 1]
                executed, carry = snapshot(executed, carry_stats[i], goals[i])
                full_paths[i] += executed
                positions[i] = executed[-1][0]
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
                elif times[i] == -1:
                    goals[i] = None
        else:  # All tasks are assigned
            for i in range(num):
                flow_time += times[i]
                path, carry = snapshot(paths[i][1:], carry_stats[i], goals[i])
                full_paths[i] += path
                add, x, y, lv = goals[i]
                if add == 0:
                    env.height[x, y] -= 1
                elif add == 1:
                    env.height[x, y] += 1
            finish = total
    return full_paths, flow_time, (generate, expand)


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.set_mirror_map()
    positions = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
    carry_stats = [True, True, True]
    num = min(arg.num, len(positions))

    profiler = cProfile.Profile()
    profiler.enable()

    if arg.start == 0:
        env.set_goal()
        env.set_shadow()
        env.set_distance_map()
        env.set_support_map()
        high_path, valids = high_lv_plan(env)
        with open('result/high_action.pkl', 'wb') as f:
            pk.dump([env.goal, high_path], f)
    else:
        with open('result/high_action.pkl', 'rb') as f:
            goal, high_path, info = pk.load(f)
            env.shadow = info['shadow']
    print(f'Number of actions: {len(high_path)}')
    print(high_path)

    paths, flow_time, stat = plan()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

    print(f'Makespan: {len(paths[0])-1}, Flow time: {flow_time}')
    print(f'Generate: {stat[0]}, Expand: {stat[1]}')
    # for t in range(len(paths[0])):
    #     print(t, [paths[i][t] for i in range(num)])

    with open('result/path.pkl', 'wb') as f:
        pk.dump([env.goal, paths], f)
