import heapq
import numpy as np
from line_profiler import LineProfiler
import cProfile

import lego
import config

'''
Valid degree:
    0: all valid locations reachable
    1: valid locations within shadow region
Heuristic mode:
    0: # of plan blocks not placed + # of scaffold blocks
    1: # of plan blocks not placed + # of scaffold blocks scaled by shadow value
    2: # of plan blocks not placed + # of scaffold blocks + 2 * # of required scaffold blocks not placed
Order mode:
    0: f -> h -> generation order
    1: f -> h -> shadow value -> generation order
    2: f -> h -> unlock number -> generation order
'''
valid_mode = 1
valid_degree = 1
heu_mode = 0
order_mode = 0

def heuristic(height, mode=0):
    """
    Calculate heuristic value for a given height map and a goal map
    Mode 0: # of plan blocks not placed + # of scaffold blocks
    Mode 1: # of plan blocks not placed + # of scaffold blocks scaled by shadow value
    Mode 2: # of plan blocks not placed + # of scaffold blocks + 2 * # of required scaffold blocks not placed
    Notes:
        Mode 1 and 2 only used for initial state
    """
    if mode == 0:
        return np.abs(height - env.goal).sum()
    elif mode == 1:
        return env.goal.sum()
    elif mode == 2:
        return env.goal.sum() + 2 * len(env.required_scaf_loc)
    else:
        raise NotImplementedError

def heuristic_diff(loc, h, add, mode=0):
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
    elif mode == 1:
        if add and scaffold:
            return 1/env.shadow_vald[(h, loc[0], loc[1])]
        elif add and not scaffold:
            return -1
        elif not add and scaffold:
            return - 1/env.shadow_vald[(h-1, loc[0], loc[1])]
        else:
            return 1
    else:
        raise NotImplementedError

def scaf_val_diff(loc, h, add):
    """
    Calculate difference of scaffold value after adding or removing a block
    scaffold value = 1 / shadow value
    """
    if add:
        scaffold = h + 1 > env.goal[loc[0], loc[1]]
    else:
        scaffold = h > env.goal[loc[0], loc[1]]

    if scaffold and add:
        return 1/env.shadow_vald[(h, loc[0], loc[1])]
    elif scaffold and not add:
        return -1/env.shadow_vald[(h-1, loc[0], loc[1])]
    else:
        return 0

def check_unlock(node):
    """Check if the action taken by the node unlocks a new plan block"""
    if node.unlock > 0:
        unlock = (node.valid_goal - node.parent.valid_goal) > 0  # unlocked goal blocks
        locs = np.transpose(np.nonzero(unlock))
        for (x, y) in locs:
            if (node.height[x, y], x, y) not in env.unlocked_loc:  # make sure the goal block is not unlocked before
                env.unlocked_loc.add((node.height[x, y], x, y))
                xs, ys = np.transpose(np.nonzero(node.height - node.parent.height))[0]
                h = node.height[xs, ys] - 1
                env.required_scaf_loc.add((h, xs, ys))
                return h, xs, ys
    return None

def update_h_vals(loc, open_list):
    """Update all h values in open list, after specifying a new required scaffold block"""
    new_open_list = []
    h, x, y = loc
    for item in open_list:
        node = item[-1]
        if node.height[x, y] < h + 1:
            node.h += 2
            push_node(new_open_list, node, mode=order_mode)
        else:
            push_node(new_open_list, node, mode=order_mode)
    return new_open_list

def push_node(open_list, node, mode=0):
    """
    Push node to open list
    Sort order (increasing):
        Mode 0: f, h, gen_id
        Mode 1: f, h, scaffold value, gen_id
        Mode 2: f, h, - # unlock, gen_id
    """
    f = node.g + node.h
    if mode == 0:
        heapq.heappush(open_list, (f, node.h, node.gen_id, node))
    elif mode == 1:
        heapq.heappush(open_list, (f, node.h, node.scaf_val, node.gen_id, node))
    elif mode == 2:
        heapq.heappush(open_list, (f, node.h, -node.unlock, node.gen_id, node))

def validate(height, new_height, loc, add):
    """
    Validate a block action:
        must exist a neighbor location with correct height, and is reachable before and after the action
    """
    new_valid = env.valid_action(new_height, mode=valid_mode, degree=valid_degree)
    x, y = loc
    h = height[x, y]
    valid = False
    if valid_mode == 0:
        for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (x2, y2) not in env.valid_loc:
                continue
            if add and height[x2, y2] == h and (x2, y2) in new_valid[0]:
                valid = True
                break
            if not add and height[x2, y2] == h-1 and (x2, y2) in new_valid[0]:
                valid = True
                break
    elif valid_mode == 1:
        for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (x2, y2) not in env.valid_loc:
                continue
            if add and height[x2, y2] == h and new_valid[0, x2, y2]:
                valid = True
                break
            if not add and height[x2, y2] == h-1 and new_valid[0, x2, y2]:
                valid = True
                break
    return valid, new_valid

def get_plan(node):
    plan = []
    while node is not None:
        plan.append(node.height)
        node = node.parent
    print(f'Number of actions: {len(plan)-1}')
    return plan[::-1]

def a_star():
    """
    A* search
    Assumptions:
        1. Node state is represented by height map
        2. Heuristic value is determined by height map
        3. Reaching a state sooner is better
    """
    open_list = []
    closed_list = dict()  # key: (height, g), value: Node

    valid = env.valid_action(env.height, mode=valid_mode, degree=valid_degree)
    root = Node(None, env.height, valid, 0, heuristic(env.height, mode=heu_mode), 0)
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

        '''Dynamic heuristic: update'''
        require_scaf = check_unlock(node)
        if require_scaf:
            open_list = update_h_vals(require_scaf, open_list)

        '''Search for child nodes'''
        if valid_mode == 0:
            for action in node.valid[1]:
                new_height = env.execute(node.height.copy(), action[:2], action[2])
                new_g = node.g + 1
                new_height_bytes = new_height.tobytes()
                # Duplicate detection: only add duplicates to open list if it has a lower g value (may add multiple)
                if new_height_bytes in closed_list and new_g >= closed_list[new_height_bytes]:
                    dup += 1
                    continue
                # Valid path detection: agent should have a way back
                valid, new_valid = validate(node.height, new_height, action[:2], action[2])
                if not valid:
                    invalid += 1
                    continue
                # Generate new node
                new_h = heuristic(new_height, mode=heu_mode)
                gen += 1
                new_node = Node(node, new_height, new_valid, new_g, new_h, gen)
                push_node(open_list, new_node, mode=order_mode)
                closed_list[new_height_bytes] = new_g
        elif valid_mode == 1:
            for (x, y) in [(x, y) for x in range(env.w) for y in range(env.w)]:
                for a in range(1, 3):
                    if node.valid[a, x, y]:
                        add = a == 1
                    else:
                        continue
                    new_height = env.execute(node.height.copy(), (x, y), add)
                    new_g = node.g + 1
                    new_height_bytes = new_height.tobytes()
                    # Duplicate detection: only add duplicates to open list if it has a lower g value (may add multiple)
                    if new_height_bytes in closed_list and new_g >= closed_list[new_height_bytes]:
                        dup += 1
                        continue
                    # Valid path detection: agent should have a way back
                    valid, new_valid = validate(node.height, new_height, (x, y), add)
                    if not valid:
                        invalid += 1
                        continue
                    # Generate new node
                    gen += 1
                    new_h = node.h + heuristic_diff((x, y), node.height[x, y], add, mode=heu_mode)
                    new_node = Node(node, new_height, new_valid, new_g, new_h, gen)
                    # scaf_val = scaf_val_diff((x, y), node.height[x, y], add)
                    # new_node.scaf_val = node.scaf_val + scaf_val
                    # new_node.scaf_val = abs(scaf_val)
                    push_node(open_list, new_node, mode=order_mode)
                    closed_list[new_height_bytes] = new_g

    raise ValueError('No solution found')

class Node:
    def __init__(self, parent, height, valid, g_val, h_val, gen_id):
        self.parent = parent
        self.height = height
        self.valid = valid
        self.g = g_val
        self.h = h_val
        self.gen_id = gen_id

        self.scaf_val = 0
        self.valid_goal = (valid[1] * (env.goal > height))
        if parent is None:
            self.unlock = 0
        else:
            self.unlock = self.valid_goal.sum() - parent.valid_goal.sum()
        # self.unlock = max(self.unlock, 0)


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)

    # lp = LineProfiler()
    # lp_wrapper = lp(env.valid_bfs_set)
    # lp_wrapper = lp(env.valid_bfs_map)
    # lp_wrapper(env.height)
    # lp_wrapper = lp(a_star)
    # lp_wrapper()
    # lp.print_stats()

    cProfile.run('a_star()', sort='tottime')

    # plan = a_star()
    # for step in plan:
    #     print(step)
