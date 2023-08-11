import pickle as pk
import cProfile
import pstats
import time

import lego
import config
from high_lv_astar import high_lv_plan
from dependency import create_graph
from cbs import cbs
from task_allocation import select_tasks, allocate_tasks, remove_tasks

"""
Hierarchical planning pipeline
1. High level plan with A*
2. Action dependency graph generation
3. Task selection and allocation
4. Low level plan with CBS
"""

def plan(dg):
    total = len(dg.nodes) if arg.select == 1 else len(dg)
    assignment = [None for _ in range(num)]
    full_paths = [[(positions[i], carry_stats[i], None)] for i in range(num)]
    finish = generate = expand = 0

    while finish < total:
        assigned = [] if arg.reselect else [t for t in assignment if t is not None]
        new_tasks, info = select_tasks(num, assigned, dg, mode=arg.select, k=arg.k)
        last_round = len(assigned) + len(new_tasks) == total - finish

        info['pos'] = positions
        info['carry'] = carry_stats
        assignment = allocate_tasks(assignment, new_tasks, info, env, arg)
        paths, times, stat = cbs(env, info, arg, last_round)
        generate += stat[0]
        expand += stat[1]
        '''Determine execution length'''
        if not last_round:  # There are tasks not assigned
            execute_t = len(paths[0])  # Execute until the assigned first task is finished
            for t in times:
                if t > 0:
                    execute_t = min(execute_t, t)
            for i in range(num):
                executed = paths[i][1:execute_t + 1]
                executed, carry = snapshot(executed, carry_stats[i], assignment[i])
                full_paths[i] += executed
                positions[i] = executed[-1][0]
                carry_stats[i] = carry
                if times[i] == -1:
                    assignment[i] = None
                elif times[i] == execute_t:
                    finish += 1
                    add, x, y, lv, _ = assignment[i]
                    if add:
                        env.height[x, y] += 1
                    else:
                        env.height[x, y] -= 1
                    remove_tasks(dg, assignment[i], mode=arg.select)
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

def hierarchy():
    """Hierarchical planning pipeline"""
    '''1. High-level plan'''
    if arg.start == 0:
        env.set_goal()
        env.set_shadow(val=True)
        env.set_distance_map()
        env.set_support_map()
        high_actions = high_lv_plan(env, arg)
        save_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
        with open(save_path, 'wb') as f:
            pk.dump([high_actions, {'goal': env.goal, 'shadow': env.shadow}], f)
    else:
        load_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
        with open(load_path, 'rb') as f:
            high_actions, info = pk.load(f)
            env.goal = info['goal']
            env.shadow = info['shadow']
            env.H = env.goal.max()
    print(f'Number of actions: {len(high_actions)}')

    '''2. Action dependency graph generation'''
    if arg.start <= 1:
        if arg.select == 0:
            dg = []
            for a in high_actions:
                dg.append((a[0], a[1], a[2], a[3], 0))
        else:
            dg = create_graph(env, high_actions, arg)
        save_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
        with open(save_path, 'wb') as f:
            pk.dump(dg, f)
    else:
        load_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
        with open(load_path, 'rb') as f:
            dg = pk.load(f)

    '''3. Task allocation and low-level path planning'''
    paths, stat = plan(dg)
    return paths, stat


if __name__ == '__main__':
    arg = config.process_config()

    env = lego.GridWorld(arg)
    env.set_mirror_map()
    if arg.teleport:
        positions = [(-1, -1) for _ in range(arg.num)]
    else:
        positions = list(env.border_loc)[:arg.num]
        positions = [(x, y, 0) for x, y in positions]
    carry_stats = [True for _ in range(arg.num)]
    num = min(arg.num, len(positions))

    profiler = cProfile.Profile()
    start = 0
    if arg.profile:
        profiler.enable()
    else:
        start = time.time()

    paths, stat = hierarchy()

    if arg.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
    else:
        print(f'Time: {time.time() - start:.3f} s')

    sum_of_cost = 0
    for path in paths:
        for t in range(len(path)):
            if path[t][0] != (-1, -1):
                sum_of_cost += 1

    print(f'Makespan: {len(paths[0])-1}, Sum of cost: {sum_of_cost}')
    print(f'Generate: {stat[0]}, Expand: {stat[1]}')

    save_path = f'result/path_{arg.map}.pkl' if arg.map > 0 else 'result/path.pkl'
    with open(save_path, 'wb') as f:
        pk.dump([env.goal, paths], f)
