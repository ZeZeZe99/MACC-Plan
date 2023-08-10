import heapq
import numpy as np
import pickle as pk
import cProfile
import pstats

import lego
import config
from scaffold_est import init_scaffold_info, update_scaffold_info, cal_goal_val

'''
Valid degree:
    0: all valid locations reachable
    1: valid locations within shadow region
    2: valid locations within shadow region, cannot remove goal blocks
Heuristic mode:
    0: # of plan blocks not placed + # of scaffold blocks
    1: # of plan blocks not placed + # of scaffold blocks + 2 * sum (goal value)
Order mode:
    0: f -> h -> generation order
'''


'''Heuristic'''
def heuristic(env, height, mode=1, new_info=None):
    """
    Calculate heuristic value for a given height map and a goal map
    Mode 0: # of plan blocks not placed + # of scaffold blocks
    Mode 1: # of plan blocks not placed + # of scaffold blocks + 2 * sum (goal value)
    Notes:
        Mode 1 and 2 only used for initial state
    """
    if mode == 0:
        return np.abs(height - env.goal).sum()
    elif mode == 1:
        # 1.0: only consider support at neighboring level (1-support)
        # return np.abs(height - env.goal).sum() + 2 * env.get_goal_val_nb(height).sum()
        # 1.1: consider support at all levels (d-support)
        # return np.abs(height - env.goal).sum() + 2 * env.get_goal_val(height).sum()
        # 1.2: consider d-support, and use goal groupings
        if new_info is None:
            return np.abs(height - env.goal).sum() + 2 * env.get_goal_val_group(height)
        else:
            return np.abs(height - env.goal).sum() + 2 * cal_goal_val(env, new_info)
    else:
        raise NotImplementedError


'''Validation'''
def validate(env, new_height, x, y, add):
    """
    Validate a block action:
        must exist a neighbor location with correct height, and is reachable before and after the action
    """
    new_valid = env.valid_bfs_map(new_height, degree=arg.valid)
    h = new_height[x, y] - 1 if add else new_height[x, y] + 1
    valid = False
    for (x2, y2) in env.valid_neighbor[x, y]:
        if add and new_height[x2, y2] == h and new_valid[0, x2, y2]:
            valid = True
            break
        if not add and new_height[x2, y2] == h-1 and new_valid[0, x2, y2]:
            valid = True
            break
    return valid, new_valid

def validate2(env, new_height, x, y, add, old_valid, mode):
    """
    Validate a block action (incremental):
        must exist a neighbor location with correct height, and is reachable before and after the action
    """
    valid, new_valid = env.update_valid_map(new_height, x, y, old_valid, degree=mode)
    '''Also invalidate bad actions'''
    if valid and ((new_height > env.goal) & np.logical_not(new_valid[0])).any():
        valid = False
    '''Check for a reachable neighbor of correct height'''
    if valid:
        valid = False
        h = new_height[x, y] - 1 if add else new_height[x, y] + 1
        for (x2, y2) in env.valid_neighbor[x, y]:
            if add and new_height[x2, y2] == h and new_valid[0, x2, y2]:
                valid = True
                break
            if not add and new_height[x2, y2] == h - 1 and new_valid[0, x2, y2]:
                valid = True
                break
    return valid, new_valid


'''Execution'''
def execute(height, loc, add):
    if add:
        height[loc] += 1
    else:
        height[loc] -= 1
    return height


'''Symmetry detection'''
def status(env, height, world, valid, h_val):
    """"""
    block = world[0]
    goal_added = world[1]
    '''Goal blocks'''
    n_goal_added = goal_added.sum()
    goal_above = env.goal > height
    n_goal_addable = (goal_above & valid[1]).sum()
    n_goal_reachable = (goal_above & valid[0]).sum()

    '''Scaffold blocks'''
    scaffold_added = world - goal_added
    n_scaffold_added = scaffold_added.sum()
    scaffold_above = (env.shadow_height > height) & np.logical_not(goal_above)
    n_scaffold_addable = (scaffold_above & valid[1]).sum()
    scaffold_below = height > env.goal
    n_scaffold_removable = (scaffold_below & valid[2]).sum()
    scaffold = scaffold_above | scaffold_below
    n_scaffold_reachable = (scaffold & valid[0]).sum()

    '''Values'''
    v_shadow_scaffold = (scaffold_added * env.shadow_val).sum()
    v_shadow_goal = (goal_added * env.shadow_val).sum()
    v_light_scaffold = (scaffold_added * env.light_val).sum()
    v_light_goal = (goal_added * env.light_val).sum()

    return (n_goal_added, n_scaffold_added, n_goal_addable, n_goal_reachable, n_scaffold_addable, n_scaffold_removable,
            n_scaffold_reachable, h_val, v_shadow_scaffold, v_light_scaffold)

def detect_symmetry(env, closed_list, g, height, world, valid, h):
    key = status(env, height, world, valid, h)
    duplicate = key in closed_list and g >= closed_list[key]
    return duplicate, key


'''Planning'''
def push_node(open_list, node, mode=0):
    """
    Push node to open list
    Sort order (increasing):
        Mode 0: f, h, gen_id
    """
    h = node.h
    f = node.g + h
    goal_added = - node.info['world'][1].sum()
    add, _, _, z = node.action
    if mode == 0:
        heapq.heappush(open_list, (f, h, node.gen_id, node))
    elif mode == 1:
        heapq.heappush(open_list, (f, h, goal_added, node.gen_id, node))
    elif mode == 2:
        heapq.heappush(open_list, (f, h, goal_added, z, node.gen_id, node))
    elif mode == 3:
        heapq.heappush(open_list, (f, h, -add, add * z, (add - 1) * z, goal_added, node.gen_id, node))
    else:
        raise NotImplementedError

def high_lv_plan(env, arg):
    """
    A* search
    Assumptions:
        1. Node state is represented by height map
        2. Heuristic value is determined by height map
        3. Reaching a state sooner is better
    """
    open_list = []
    closed_list = dict()  # key: (height, g), value: Node
    closed_stat = dict()

    valid = env.valid_bfs_map(env.height, degree=arg.valid)
    root_info = init_scaffold_info(env, env.height)
    root_h = heuristic(env, env.height, mode=arg.high_heu, new_info=root_info)
    root = Node(None, env.height, valid, 0, root_h, 0, root_info)
    push_node(open_list, root,  mode=arg.high_order)
    gen = expand = invalid = dup = dup2 = 0

    closed_list[root.height.tobytes()] = 0
    if arg.symmetry == 1:
        stat = status(env, env.height, root_info['world'], valid, root_h)
        closed_stat[stat] = 0

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        # Duplicate detection: skip if the node has been expanded with a lower g value
        # if node.height.tobytes() in closed_list and node.g > closed_list[node.height.tobytes()]:
        #     dup2 += 1
        #     continue
        expand += 1

        '''Completion check'''
        if np.array_equal(node.height, env.goal):
            print(f'Generated: {gen}, Expanded: {expand}, Invalid: {invalid}, Duplicate: {dup}, Duplicate2: {dup2}')
            return get_plan(node)

        '''Search for child nodes'''
        for a in range(1, 3):
            add = a == 1
            for (x, y) in [(x, y) for x in range(env.w) for y in range(env.w)]:
                '''1st round action validation'''
                if not node.valid[a, x, y]:
                    continue
                # Skip removing goal blocks
                if arg.valid == 2 and not add and node.height[x, y] <= env.goal[x, y]:
                    continue

                '''Execute action'''
                new_g = node.g + 1
                new_height = execute(node.height.copy(), (x, y), add)
                z = new_height[x, y] - 1 if add else new_height[x, y]
                new_world = node.info['world'].copy()
                is_goal = env.goal3d[z, x, y] == 1
                new_world[0, z, x, y] = add
                new_world[1, z, x, y] = is_goal * add

                '''Duplicate detection'''
                key = new_height.tobytes()
                if key in closed_list and closed_list[key] <= new_g:
                    dup += 1
                    continue

                '''2nd round action validation: agent should have a way back'''
                valid, new_valid = validate2(env, new_height, x, y, add, node.valid, arg.valid)
                if not valid:
                    invalid += 1
                    continue
                closed_list[key] = new_g  # Save g value for duplicate detection

                '''New heuristic value'''
                valid, new_info = update_scaffold_info(env, node.info, add, (x, y), z, new_world)
                if not valid:
                    invalid += 1
                    continue
                new_h = heuristic(env, new_height, mode=arg.high_heu, new_info=new_info)

                '''Symmetry detection'''
                if arg.symmetry == 1:
                    duplicate, key = detect_symmetry(env, closed_stat, new_g, new_height, new_world, new_valid, new_h)
                    if duplicate:
                        dup2 += 1
                        continue
                    closed_stat[key] = new_g

                '''Generate child node'''
                gen += 1
                action = (int(add), x, y, z)
                new_node = Node(node, new_height, new_valid, new_g, new_h, gen, new_info, action)
                push_node(open_list, new_node, mode=arg.high_order)

    raise ValueError('No solution found')

def get_plan(node):
    heights, actions = [], []
    while node is not None:
        heights.append(node.height)
        node = node.parent
    heights.reverse()
    for i in range(len(heights) - 1):
        diff = heights[i+1] - heights[i]
        x, y = np.argwhere(diff)[0]
        h1, h2 = heights[i][x, y], heights[i+1][x, y]
        add = h2 > h1
        lv = min(h1, h2)
        actions.append((int(add), x, y, lv))
    return actions

class Node:
    def __init__(self, parent, height, valid, g_val, h_val, gen_id, info, action=(-1, -1, -1, -1)):
        self.parent = parent
        self.height = height
        self.valid = valid
        self.g = g_val
        self.h = h_val
        self.gen_id = gen_id
        self.info = info
        self.action = action


if __name__ == '__main__':
    arg = config.process_config()

    env = lego.GridWorld(arg)
    env.set_goal()
    env.set_shadow(val=True)
    env.set_distance_map()
    env.set_support_map()
    env.set_light()

    profiler = cProfile.Profile()
    profiler.enable()
    high_actions = high_lv_plan(env, arg)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    print(f'Number of actions: {len(high_actions)}')
    print(high_actions)
    save_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
    with open(save_path, 'wb') as f:
        pk.dump([high_actions, {'goal': env.goal, 'shadow': env.shadow}], f)
