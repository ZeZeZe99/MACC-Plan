import heapq
import numpy as np
import pickle as pk
import cProfile
import pstats

import lego
import config

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


'''Heuristic'''
def heuristic(env, height, mode=0):
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
        return np.abs(height - env.goal).sum() + 2 * env.get_goal_val_group(height).sum()
    else:
        raise NotImplementedError

def heuristic_diff(env, loc, h, add, mode=0):
    """Calculate difference of heuristic value after adding or removing a block"""
    if add:
        scaffold = h + 1 > env.goal[loc[0], loc[1]]
    else:
        scaffold = h > env.goal[loc[0], loc[1]]

    if mode == 0:
        if add and scaffold:
            return 1
        elif add and not scaffold:
            return -1
        elif not add and scaffold:
            return -1
        else:
            return 1
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


'''Planning'''
def push_node(open_list, node, mode=0):
    """
    Push node to open list
    Sort order (increasing):
        Mode 0: f, h, gen_id
    """
    f = node.g + node.h
    if mode == 0:
        heapq.heappush(open_list, (f, node.h, node.gen_id, node))
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

    valid = env.valid_bfs_map(env.height, degree=valid_degree)
    root = Node(None, env.height, valid, 0, heuristic(env, env.height, mode=heu_mode), 0)
    push_node(open_list, root,  mode=order_mode)
    closed_list[root.height.tobytes()] = 0
    gen = expand = invalid = dup = dup2 = 0

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
                '''Validate action'''
                if not node.valid[a, x, y]:
                    continue
                # Skip removing goal blocks
                if valid_degree == 2 and not add and node.height[x, y] <= env.goal[x, y]:
                    continue
                # Skip adding a scaffold twice
                if valid_degree == 3 and add and node.height[x, y] >= env.goal[x, y]:
                    if (node.height[x, y], x, y) in node.added_scaffold:
                        continue
                new_height = execute(node.height.copy(), (x, y), add)
                new_g = node.g + 1
                new_height_bytes = new_height.tobytes()
                # Duplicate detection: only add duplicates to open list if it has a lower g value (may add multiple)
                if new_height_bytes in closed_list and new_g >= closed_list[new_height_bytes]:
                    dup += 1
                    continue
                # Valid path detection: agent should have a way back
                # valid, new_valid = validate(env, new_height, x, y, add)
                valid, new_valid = validate2(env, new_height, x, y, add, node.valid)
                if not valid:
                    invalid += 1
                    continue
                '''Generate child node'''
                gen += 1
                # new_h = node.h + heuristic_diff(env, (x, y), node.height[x, y], add, mode=heu_mode)
                new_h = heuristic(env, new_height, mode=heu_mode)
                new_node = Node(node, new_height, new_valid, new_g, new_h, gen)
                # Mark added scaffold (can only add once)
                # if valid_degree == 3 and add and new_height[x, y] > env.goal[x, y]:
                #     new_node.added_scaffold.add((new_height[x, y] - 1, x, y))

                push_node(open_list, new_node, mode=order_mode)
                closed_list[new_height_bytes] = new_g

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
    def __init__(self, parent, height, valid, g_val, h_val, gen_id):
        self.parent = parent
        self.height = height
        self.valid = valid
        self.g = g_val
        self.h = h_val
        self.gen_id = gen_id


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.set_goal()
    env.set_shadow()
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
