import pickle as pk
import cProfile
import pstats
import time

import lego
import config
from high_lv_astar import high_lv_plan
from dependency import create_graph
from level_graph import reorder_level
from plan_priority import priority_plan
# from plan_sequence import plan

"""
Hierarchical planning pipeline
1. High level plan with A*
2. Action reordering
3. Task selection and allocation
4. Low level plan with CBS
"""

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

    '''2. Action sequence reordering'''
    if arg.start <= 1:
        if arg.select == 0:
            all_tasks = []
            for a in high_actions:
                all_tasks.append((a[0], a[1], a[2], a[3], 0))
        elif arg.select == 1:
            all_tasks = reorder_level(env, high_actions, arg)
        else:
            all_tasks = create_graph(env, high_actions, arg)
        save_path = f'result/tasks_{arg.map}.pkl' if arg.map > 0 else 'result/tasks.pkl'
        with open(save_path, 'wb') as f:
            pk.dump(all_tasks, f)
    else:
        load_path = f'result/tasks_{arg.map}.pkl' if arg.map > 0 else 'result/tasks.pkl'
        with open(load_path, 'rb') as f:
            all_tasks = pk.load(f)

    '''3. Task allocation and low-level path planning'''
    # paths, stat = plan(env, arg, all_tasks, positions, carry_stats)
    paths, stat = priority_plan(env, all_tasks, arg)
    return paths, stat


if __name__ == '__main__':
    arg = config.process_config()

    env = lego.GridWorld(arg)
    env.set_mirror_map()
    # if arg.teleport:
    #     positions = [(-1, -1) for _ in range(arg.num)]
    # else:
    #     positions = list(env.border_loc)[:arg.num]
    #     positions = [(x, y, 0) for x, y in positions]
    # carry_stats = [True for _ in range(arg.num)]
    # num = arg.num

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
    # print(f'Generate: {stat[0]}, Expand: {stat[1]}')

    save_path = f'result/path_{arg.map}.pkl' if arg.map > 0 else 'result/path.pkl'
    with open(save_path, 'wb') as f:
        pk.dump([env.goal, paths], f)
