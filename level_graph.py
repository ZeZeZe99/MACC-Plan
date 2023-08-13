import numpy as np
import pickle as pk
import cProfile
import pstats

import lego
import config
from high_lv_astar import validate2, execute

"""
Re-order action sequence for better parallelization
Conservatively, break the re-ordered action sequence into levels
It is guaranteed to be feasible to execute the reordered action sequence if:
    1. Execute levels in order
    2. Execute actions within each level in order
"""

def precedence(env, a1, a2, height, valid, mode):
    """
    Action 1 and action 2 have precedence constraint w.r.t. a height map if swapping them leads to error, e.g.,
        1) Action 1 and 2 are at the same x-y location, or
        2) Action 2 cannot be executed before action 1, or
        3) Action 1 cannot be executed after action 2
    Height is a snapshot when both actions are not executed yet
    """
    add1, x1, y1, z1, _ = a1
    add2, x2, y2, z2, _ = a2

    '''Check if executing action 2 before action 1 leads to error'''
    if not valid[2 - add2, x2, y2]:  # Action 2 is not in the valid map
        return True
    height_a2 = execute(height.copy(), (x2, y2), add2)
    valid2, valid_a2 = validate2(env, height_a2, x2, y2, add2, valid, mode)
    if not valid2:  # Does not exist a path back to border
        return True
    '''Check if executing action 1 after action 2 leads to error'''
    if not valid_a2[2 - add1, x1, y1]:  # Action 1 is not in the valid map
        return True
    height_a1 = execute(height_a2, (x1, y1), add1)
    valid1 = validate2(env, height_a1, x1, y1, add1, valid_a2, mode)[0]
    return not valid1

def precedence_level(env, lv_actions, lv_height, lv_valid, new_node):
    """
    Check if the new action can be performed before all the actions in the current level
    Conservatively, check if current level can be performed in order after the new action
    """
    x, y = new_node[1], new_node[2]
    for a in lv_actions:
        if x == a[1] and y == a[2]:
            return True
    for i, a in enumerate(lv_actions):
        if precedence(env, a, new_node, lv_height[i], lv_valid[i], mode=1):
            return True
    return False

def precedence_pairwise(env, lv_actions, lv_block, new_node):
    add2, x2, y2, z2, _ = new_node
    for a in lv_actions:
        add1, x1, y1, z1, _ = a
        '''node 1 and node 2 are at the same x-y location'''
        if x1 == x2 and y1 == y2:
            return True
        '''node 1 and node 2 are at neighboring x-y locations'''
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            continue
        if add2 == 1:  # node 2 adds a block
            if z1 == z2:
                '''Add node 2 may make node 1 invalid'''
                c = 0
                for nbrx, nbry in env.valid_neighbor[(x1, y1)]:  # Look at node 1's neighbors
                    if nbrx == x2 and nbry == y2:
                        c += 1
                    elif lv_block[z1, nbrx, nbry]:  # Neighbor may have a block added
                        c += 1
                    elif z1 > 0 and not lv_block[z1 - 1, nbrx, nbry]:  # Node 1 cannot be performed from neighbor
                        c += 1
                if c == 4:
                    return True
        else:  # node 2 removes a block
            if z1 == z2 and add1 == 0:
                '''node 2 may not be able to be removed before node 1'''
                c = 0
                for nbrx, nbry in env.valid_neighbor[(x2, y2)]:  # Look at node 2's neighbors
                    if lv_block[z1, nbrx, nbry]:
                        c += 1
                    elif z1 > 0 and not lv_block[z1 - 1, nbrx, nbry]:
                        c += 1
                if c == 4:
                    return True
    return False

def reorder_level(env, actions, arg):
    """Create a dependency graph for the given actions"""
    nodes = []

    heights = np.zeros((len(actions) + 1, *env.world_shape), dtype=np.int8)
    action_count = np.zeros((env.goal.max(), *env.world_shape), dtype=np.int8)

    lv_heights = []
    lv_valids = []
    lv_blocks = []
    lv_actions = {}
    highest_lv = -1

    for i in range(len(actions)):
        add, x, y, z = actions[i]
        new_node = (add, x, y, z, action_count[z, x, y])
        action_count[z, x, y] += 1
        nodes.append(new_node)
        new_lv = 0

        '''1. Iteratively check precedence constraint with each level, starting from the highest level'''
        for lv in range(highest_lv, -1, -1):
            if precedence_level(env, lv_actions[lv], lv_heights[lv], lv_valids[lv], new_node):
                new_lv = lv + 1
                break
            if precedence_pairwise(env, lv_actions[lv], lv_blocks[lv], new_node):
                new_lv = lv + 1
                break

        if new_lv not in lv_actions:
            lv_actions[new_lv] = [new_node]
            lv_heights.append([heights[i]])
            lv_valids.append([env.valid_bfs_map(heights[i], degree=1)])
            lv_block = action_count % 2 == 1
            lv_block[z, x, y] = 1
            lv_blocks.append(lv_block)
        else:
            height = lv_heights[new_lv][-1].copy()
            a = lv_actions[new_lv][-1]
            execute(height, a[1:3], a[0])
            lv_actions[new_lv].append(new_node)
            lv_heights[new_lv].append(height)
            lv_valids[new_lv].append(env.valid_bfs_map(height, degree=1))
            lv_blocks[new_lv][z, x, y] = 1
            for lv in range(new_lv + 1, highest_lv + 1):
                lv_valids[lv] = []
                for j in range(len(lv_heights[lv])):
                    execute(lv_heights[lv][j], (x, y), add)
                    lv_valids[lv].append(env.valid_bfs_map(lv_heights[lv][j], degree=1))
                lv_blocks[lv][z, x, y] = add
        highest_lv = max(highest_lv, new_lv)

        '''Update records'''
        heights[i + 1:, x, y] += 1 if add == 1 else -1

    '''Level results'''
    # lv_result = []
    # for lv in range(1, highest_lv + 1):
    #     lv_result.append(lv_heights[lv][0])
    # lv_result.append(env.goal)
    # lv_result = np.stack(lv_result, axis=0)

    return lv_actions


if __name__ == '__main__':
    arg = config.process_config()

    env = lego.GridWorld(arg)
    load_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
    with open(load_path, 'rb') as f:
        high_actions, info = pk.load(f)
    env.goal = info['goal']
    env.shadow = info['shadow']
    env.H = env.goal.max()

    profiler = cProfile.Profile()
    profiler.enable()

    levels = reorder_level(env, high_actions, arg)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)

    save_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
    with open(save_path, 'wb') as f:
        pk.dump(levels, f)
