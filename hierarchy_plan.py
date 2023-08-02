import pickle as pk
import cProfile
import pstats

import lego
import config
from high_lv_astar import high_lv_plan
from dependency import create_graph
from cbs import cbs
from task_allocation import allocate


allocate_mode = 1

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

def plan():
    if allocate_mode == 0:
        todo = high_actions.copy()
        total = len(todo)
    elif allocate_mode == 1:
        dg.remove_node('S')
        todo = dg
        total = len(dg.nodes)
    else:
        raise NotImplementedError

    assignment = [None for _ in range(num)]
    full_paths = [[(positions[i], carry_stats[i], None)] for i in range(num)]
    finish = generate = expand = 0

    while finish < total:
        assignment, num_dummy = allocate(num, todo, assignment, mode=allocate_mode)
        paths, times, stat = cbs(env, assignment, positions, carry_stats)
        generate += stat[0]
        expand += stat[1]
        '''Determine execution length'''
        if num - num_dummy < total - finish:  # There are tasks not assigned
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
                if times[i] == execute_t:
                    finish += 1
                    add, x, y, lv, _ = assignment[i]
                    if add:
                        env.height[x, y] += 1
                    else:
                        env.height[x, y] -= 1
                    if allocate_mode == 0:
                        todo.remove(assignment[i])
                    else:
                        todo.remove_node(assignment[i])
                    assignment[i] = None
                elif times[i] == -1:
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

    if arg.start == 0:  # Generate high-level plan
        env.set_goal()
        env.set_shadow(val=True)
        env.set_distance_map()
        env.set_support_map()
        high_actions = high_lv_plan(env)
        save_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
        with open(save_path, 'wb') as f:
            pk.dump([high_actions, {'goal': env.goal, 'shadow': env.shadow}], f)
        print(f'Number of actions: {len(high_actions)}')
        dg = create_graph(env, high_actions)
        save_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
        with open(save_path, 'wb') as f:
            pk.dump(dg, f)

    elif arg.start == 1:  # Load high-level plan
        load_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
        with open(load_path, 'rb') as f:
            high_actions, info = pk.load(f)
            env.goal = info['goal']
            env.shadow = info['shadow']
            env.H = env.goal.max()
        print(f'Number of actions: {len(high_actions)}')
        dg = create_graph(env, high_actions)
        save_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
        with open(save_path, 'wb') as f:
            pk.dump(dg, f)

    else:  # Load dependency graph
        load_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
        with open(load_path, 'rb') as f:
            dg = pk.load(f)

    paths, stat = plan()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

    sum_of_cost = 0
    for path in paths:
        for t in range(len(path) - 1, -1, -1):
            if path[t][2] is not None:
                sum_of_cost += t
                break
    fuel = 0
    for path in paths:
        for t in range(len(path) - 1):
            if path[t][0] != path[t + 1][0] or path[t][1] != path[t + 1][1]:
                fuel += 1

    print(f'Makespan: {len(paths[0])-1}, Sum of cost: {sum_of_cost}, Fuel: {fuel}')
    print(f'Generate: {stat[0]}, Expand: {stat[1]}')

    save_path = f'result/path_{arg.map}.pkl' if arg.map > 0 else 'result/path.pkl'
    with open(save_path, 'wb') as f:
        pk.dump([env.goal, paths], f)
