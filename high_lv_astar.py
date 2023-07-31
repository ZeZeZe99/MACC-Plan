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
    3: valid locations within shadow region, cannot place a scaffold block twice
Heuristic mode:
    0: # of plan blocks not placed + # of scaffold blocks
    1: # of plan blocks not placed + # of scaffold blocks + 2 * sum (goal value)
Order mode:
    0: f -> h -> generation order
'''
valid_degree = 2
heu_mode = 1
order_mode = 0
dup_mode = 0


'''Heuristic'''
def heuristic(env, height, mode=0, new_info=None):
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

def heuristic_diff(add, loc, goal, new_height, new_info, mode=0):
    """Calculate difference of heuristic value after adding or removing a block"""
    h = new_height[loc]
    if add:
        scaffold = h > goal[loc]
    else:
        scaffold = h >= goal[loc]

    if mode == 0:
        if add and scaffold:
            return 1
        elif add and not scaffold:
            return -1
        elif not add and scaffold:
            return -1
        else:
            return 1
    elif mode == 1:
        if add and scaffold:
            h_diff = 1
        elif add and not scaffold:
            h_diff = -1
        elif not add and scaffold:
            h_diff = -1
        else:
            h_diff = 1
        h_diff += 2 * env.update_goal_val(new_info)
    else:
        raise NotImplementedError


'''Validation'''
def validate(env, new_height, x, y, add):
    """
    Validate a block action:
        must exist a neighbor location with correct height, and is reachable before and after the action
    """
    new_valid = env.valid_bfs_map(new_height, degree=valid_degree)
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

def validate2(env, new_height, x, y, add, old_valid):
    """
    Validate a block action (incremental):
        must exist a neighbor location with correct height, and is reachable before and after the action
    """
    valid, new_valid = env.update_valid_map(new_height, x, y, old_valid, degree=valid_degree)
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
def transform(matrix, mode):
    """8 transformations of a 2D matrix. Last 4 only used for square matrix"""
    if mode == 0:
        return matrix
    if mode == 1:
        return np.rot90(matrix, 2)
    if mode == 2:
        return np.flipud(matrix)
    if mode == 3:
        return np.fliplr(matrix)
    if mode == 4:
        return np.rot90(matrix, 1)
    if mode == 5:
        return np.rot90(matrix, 3)
    if mode == 6:
        return np.rot90(np.flipud(matrix), 1)
    else:
        return np.rot90(np.flipud(matrix), 3)

def status(env, height, world, valid):
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
    shadow_added = scaffold_added * env.shadow_val
    n_shadow_added = shadow_added.sum()
    return (n_goal_added, n_scaffold_added, n_shadow_added, n_goal_addable, n_goal_reachable,
            n_scaffold_addable, n_scaffold_removable)

def detect_duplicate(env, closed_list, g, height, mode=0, world=None, valid=None):
    if mode == 0:
        key = height.tobytes()
        duplicate = key in closed_list and g >= closed_list[key]
        return duplicate, key
    elif mode == 1:
        matrix = (height - env.goal)[env.box]
        for t in range(8 if env.square_box else 4):
            key = transform(matrix, t).tobytes()
            if key in closed_list and g >= closed_list[key]:
                return True, key
        return False, matrix.tobytes()
    elif mode == 2:
        key = status(env, height, world, valid)
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
    if mode == 0:
        heapq.heappush(open_list, (f, h, node.gen_id, node))
    else:
        raise NotImplementedError

def high_lv_plan(env):
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

    valid = env.valid_bfs_map(env.height, degree=valid_degree)
    root_info = init_scaffold_info(env, env.height)
    stat = status(env, env.height, root_info['world'], valid)
    root_h = heuristic(env, env.height, mode=heu_mode, new_info=root_info)
    root = Node(None, env.height, valid, 0, root_h, 0, root_info)
    push_node(open_list, root,  mode=order_mode)
    gen = expand = invalid = dup = dup2 = 0

    if dup_mode == 0:
        closed_list[root.height.tobytes()] = 0
    elif dup_mode == 2:
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
                if valid_degree == 2 and not add and node.height[x, y] <= env.goal[x, y]:
                    continue

                '''Execute action'''
                new_g = node.g + 1
                new_height = execute(node.height.copy(), (x, y), add)
                z = new_height[x, y] - 1 if add else new_height[x, y]
                new_world = node.info['world'].copy()
                is_goal = env.goal3d[z, x, y] == 1
                new_world[0, z, x, y] = add
                new_world[0, z, x, y] = is_goal * add

                '''1st round duplicate detection'''
                if dup_mode == 0:
                    duplicate, key = detect_duplicate(env, closed_list, new_g, new_height, mode=0)
                    if duplicate:
                        dup += 1
                        continue

                '''2nd round action validation: agent should have a way back'''
                valid, new_valid = validate2(env, new_height, x, y, add, node.valid)
                if not valid:
                    invalid += 1
                    continue

                '''2nd round duplicate detection'''
                if dup_mode == 2:
                    duplicate, key = detect_duplicate(env, closed_list, new_g, new_height,
                                                      mode=2, world=new_world, valid=new_valid)

                '''Generate child node'''
                gen += 1
                new_info = update_scaffold_info(env, node.info, add, (x, y), z, new_world)
                new_h = heuristic(env, new_height, mode=heu_mode, new_info=new_info)
                new_node = Node(node, new_height, new_valid, new_g, new_h, gen, new_info)

                push_node(open_list, new_node, mode=order_mode)
                closed_list[key] = new_g

    raise ValueError('No solution found')

def get_plan(node):
    heights, valids, actions = [], [], []
    while node is not None:
        heights.append(node.height)
        valids.append(node.valid)
        node = node.parent
    heights.reverse()
    valids.reverse()
    for i in range(len(heights) - 1):
        diff = heights[i+1] - heights[i]
        x, y = np.argwhere(diff)[0]
        h1, h2 = heights[i][x, y], heights[i+1][x, y]
        add = h2 > h1
        lv = min(h1, h2)
        actions.append((int(add), x, y, lv))
    return actions, valids

class Node:
    def __init__(self, parent, height, valid, g_val, h_val, gen_id, info):
        self.parent = parent
        self.height = height
        self.valid = valid
        self.g = g_val
        self.h = h_val
        self.gen_id = gen_id
        self.info = info


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.set_goal()
    env.set_shadow(val=True)
    env.set_distance_map()
    env.set_support_map()

    # lp = LineProfiler()
    # lp_wrapper = lp(high_lv_plan)
    # lp_wrapper(env)
    # lp.print_stats()

    profiler = cProfile.Profile()
    profiler.enable()
    high_actions, valids = high_lv_plan(env)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    print(f'Number of actions: {len(high_actions)}')
    print(high_actions)
    with open('result/high_action.pkl', 'wb') as f:
        pk.dump([env.goal, high_actions, {'valid': valids, 'shadow': env.shadow}], f)