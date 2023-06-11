import heapq
import numpy as np
from line_profiler import LineProfiler
import cProfile

import lego
import config

valid_mode = 1
valid_degree = 1

def heuristic(height, mode=0):
    """
    Calculate heuristic value for a given height map
    Mode 0: # of plan blocks not placed + # of scaffold blocks
    """
    if mode == 0:
        return np.abs(height - env.goal).sum()
    else:
        raise NotImplementedError

def push_node(open_list, node):
    """
    Push node to open list
    Sort order: f (lower), h (lower), gen_id (lower)
    """
    heapq.heappush(open_list, (node.f, node.h, node.gen_id, node))

def get_plan(node):
    plan = []
    while node is not None:
        plan.append(node.height)
        node = node.parent
    return plan[::-1]

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

    root = Node(None, env.height, 0, heuristic(env.height), 0)
    root.valid = env.valid_action(root.height, mode=valid_mode, degree=valid_degree)
    push_node(open_list, root)
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
                new_h = heuristic(new_height)
                gen += 1
                new_node = Node(node, new_height, new_g, new_h, gen)
                new_node.valid = new_valid
                push_node(open_list, new_node)
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
                    new_h = heuristic(new_height)
                    gen += 1
                    new_node = Node(node, new_height, new_g, new_h, gen)
                    new_node.valid = new_valid
                    push_node(open_list, new_node)
                    closed_list[new_height_bytes] = new_g

    raise ValueError('No solution found')

class Node:
    def __init__(self, parent, height, g_val, h_val, gen_id):
        self.parent = parent
        self.height = height
        self.g = g_val
        self.h = h_val
        self.f = self.g + self.h
        self.gen_id = gen_id


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
