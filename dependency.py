import networkx as nx
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import cProfile
import pstats

import lego
import config
from high_lv_astar import validate2, execute

def undo(height, action):
    """Undo the action on the height map"""
    if action[0] == 1:
        height[action[1], action[2]] -= 1
    else:
        height[action[1], action[2]] += 1
    return height

def precedence(env, a1, a2, height, valid):
    """
    Action 1 and action 2 have precedence constraint if swapping them leads to error, e.g.,
        1) Action 1 and 2 are at the same x-y location, or
        2) Action 2 cannot be executed before action 1, or
        3) Action 1 cannot be executed after action 2
    Height is a snapshot when both actions are not executed yet
    """
    add1, x1, y1, z1, _ = a1
    add2, x2, y2, z2, _ = a2

    '''a1 and a2 are at the same x-y location'''
    if x1 == x2 and y1 == y2:
        return True
    '''Check if executing action 2 before action 1 leads to error'''
    if not valid[2 - add2, x2, y2]:  # Action 2 is not in the valid map
        return True
    height_a2 = execute(height.copy(), (x2, y2), add2)
    valid2, valid_a2 = validate2(env, height_a2, x2, y2, add2, valid)
    if not valid2:  # Does not exist a path back to border
        return True
    '''Check if executing action 1 after action 2 leads to error'''
    if not valid_a2[2 - add1, x1, y1]:  # Action 1 is not in the valid map
        return True
    height_a1 = execute(height_a2, (x1, y1), add1)
    valid1 = validate2(env, height_a1, x1, y1, add1, valid_a2)[0]
    return not valid1

def create_graph(env, actions):
    """Create a dependency graph for the given actions"""
    g = nx.DiGraph()
    nodes = []

    heights = np.zeros((len(actions) + 1, *env.world_shape), dtype=np.int8)
    valids = np.zeros((len(actions), 3, *env.world_shape), dtype=np.int8)
    count = np.zeros((env.goal.max(), *env.world_shape), dtype=np.int8)

    for i in range(len(actions)):
        add, x, y, z = actions[i]
        new_node = (add, x, y, z, count[z, x, y])
        count[z, x, y] += 1
        nodes.append(new_node)
        no_precedence = True

        valids[i] = env.valid_bfs_map(heights[i], degree=1)
        '''1. Find precedence constraint with actions before i in reverse order (early stop)'''
        ptr = i - 1
        for j in range(i - 1, -1, -1):
            # Height snapshot: right before j is executed
            if precedence(env, nodes[j], nodes[i], heights[j], valids[j]):
                g.add_edge(nodes[j], nodes[i])
                no_precedence = False
                ptr = j
                break

        '''2. Recursively find precedence constraints, starting with all leaf nodes the pointer'''
        leaves = [j for j in range(ptr) if g.out_degree[nodes[j]] == 0]
        leaf_height = heights[i].copy()
        temp_g = g.copy()
        while len(leaves) > 0:
            next_height = leaf_height.copy()
            for j in leaves:
                # Height snapshot: right before i is executed, plus j is undone
                height = undo(leaf_height.copy(), actions[j])
                valid = env.valid_bfs_map(height, degree=1)
                # If precedence constraint exists, j is not a leaf node anymore
                if precedence(env, nodes[j], nodes[i], height, valid):
                    g.add_edge(nodes[j], nodes[i])
                    no_precedence = False
                    temp_g.add_edge(nodes[j], nodes[i])
                # Otherwise, j is still a leaf node, but need to check with predecessors of j
                else:
                    temp_g.remove_node(nodes[j])
                    next_height = undo(next_height, actions[j])
            leaves = [j for j in range(ptr) if nodes[j] in temp_g.nodes and temp_g.out_degree[nodes[j]] == 0]
            leaf_height = next_height

        '''3. No precedence constraint, connect to dummy source node'''
        if no_precedence:
            g.add_edge('S', new_node)

        heights[i + 1:, x, y] += 1 if add == 1 else -1  # Update height map

    return g


def draw_graph(graph):
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='skyblue', edge_color='black',
            width=1.5, alpha=0.7)
    plt.show()


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    load_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
    with open(load_path, 'rb') as f:
        high_actions, info = pk.load(f)
    env.goal = info['goal']
    env.shadow = info['shadow']
    env.H = env.goal.max()

    profiler = cProfile.Profile()
    profiler.enable()

    g = create_graph(env, high_actions)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)

    draw_graph(g)

    save_path = f'result/dependency_{arg.map}.pkl' if arg.map > 0 else 'result/dependency.pkl'
    with open(save_path, 'wb') as f:
        pk.dump(g, f)
