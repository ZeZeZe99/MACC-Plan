import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import heapq
from itertools import count
import time
import A_star as astar
import A_star_v2 as astar_planner

class ActionDependencyGraph():
    def __init__(self, action_sequence):
        self.action_sequence = action_sequence
        self.CPU_start_time = time.time()
        self.adg = nx.DiGraph()
        self.init_graph()
        self.add_virtual_nodes()
        self.draw_graph()
    
    def init_graph(self):
        for i in range(len(self.action_sequence)-1):
            [x], [y] = np.nonzero(self.action_sequence[i+1]-self.action_sequence[i])
            z = self.action_sequence[i][x][y]
            operation = np.sum(self.action_sequence[i+1]-self.action_sequence[i])
            self.adg.add_node((x, y, z, operation))
            self.add_dependencies(x, y, z, operation, i)
    
    def add_dependencies(self, x, y, z, operation, i):
        world_state = self.action_sequence[i]
        all_leaf_nodes = [node for node in self.adg.nodes() if len(self.adg.succ[node]) == 0]
        for node in self.adg:
            # is node an operation on (x, y)?
            if node[0] == x \
                and node[1] == y \
                and (node[2] == world_state[x][y]-1 or node[2] == world_state[x][y]+1) \
                and node[3]==operation:
                self.adg.add_edge(node, (x, y, z, operation))
            # is node an operation on a neighbour of (x, y)?
            elif self.is_a_neighbour((x, y), node[0:2]) and node[2] == world_state[node[0]][node[1]]-1:
                self.adg.add_edge(node, (x, y, z, operation))
            # is node a leaf node?
            if node in all_leaf_nodes and node!=(x, y, z, operation):
                if operation==-1:
                    for path in nx.all_simple_paths(self.adg, (x, y, z-1, 1), node):
                        if len(path) > 0:
                            self.adg.add_edge(node, (x, y, z, operation))
                elif operation ==1:
                    # check if node exists in graph
                    if (x, y, z+1, -1) in self.adg.nodes:
                        for path in nx.all_simple_paths(self.adg, (x, y, z+1, -1), node):
                            if len(path) > 0:
                                self.adg.add_edge(node, (x, y, z, operation))
        
    def add_virtual_nodes(self):
        # add a virtual start node to connect all subgraphs
        self.adg.add_node((-1, -1, -1, 0))
        for node in self.adg.nodes:
            if len(self.adg.pred[node]) == 0 and node != (-1, -1, -1, 0):
                self.adg.add_edge((-1, -1, -1, 0), node)
    
    def is_a_neighbour(self, a, b):        
        if a[0] == b[0] and a[1] == b[1]+1:
            return True
        elif a[0] == b[0] and a[1] == b[1]-1:
            return True
        elif a[0] == b[0]+1 and a[1] == b[1]:
            return True
        elif a[0] == b[0]-1 and a[1] == b[1]:
            return True
        else:
            return False
    
    def draw_graph(self):
        pos = nx.shell_layout(self.adg)
        # pos = nx.spring_layout(self.adg)
        print("Time to build Dependency Graph", time.time()-self.CPU_start_time)
        nx.draw(self.adg, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', edge_color='black', width=1.5, alpha=0.7)
        # plt.show()
    
    # def get_all_dependencies(self, node):
    #     dependencies = []
    #     for path in nx.all_simple_paths(self.adg, (-1, -1, -1, 0), node):
    #         for n in path:
    #             if n not in dependencies:
    #                 dependencies.append(n)
    #     return dependencies

def main():
    # action_sequence = astar.main()
    action_sequence = astar_planner.main()
    graph = ActionDependencyGraph(action_sequence)
    return graph

if __name__ == '__main__':
    main()