import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from goal import *

class MinSpanningTree:
    def __init__(self, struct):
        self.structure = struct
        # self.structure = self.compute_workspace(struct)
        self.edges = self.get_min_spanning_tree()

    def compute_workspace(self, structure):
        borders = [0, 0, 0, 0] # top(0), bottom(1), left(2), right(3)
        for i in range(len(structure)):
            for j in range(len(structure)):
                borders[0] = max(borders[0], structure[i, j]-j+1)
                borders[1] = max(borders[1], structure[i, j]-(len(structure)-j+1))
                borders[2] = max(borders[2], structure[i, j]-i+1)
                borders[3] = max(borders[3], structure[i, j]-(len(structure)-i+1))
        goal = np.zeros((len(structure)+borders[0]+borders[1], len(structure)+borders[2]+borders[3]))
        for i in range(len(structure)):
            for j in range(len(structure)):
                goal[i+borders[0], j+borders[2]] = structure[i, j]
        return goal

    def usefulness_matrix(self, map):
        u = np.zeros((len(map), len(map[0])))
        self.useful_tree = nx.DiGraph()
        self.max_scaffolding = u.copy()
        for i in range(0, len(map)):
            for j in range(0, len(map[0])):
                if map[i][j]>0:
                    self.useful_tree.add_node((i,j, map[i][j], 1))
                    # print("adding node", (i,j, map[i][j]))
                    for x in range(i - map[i][j]-1, i+ map[i][j]+1):
                        for y in range(j - map[i][j]-1, j+ map[i][j]+1):
                            if (abs(x-i)+abs(y-j))<map[i][j]:
                                self.max_scaffolding[x][y]=max(self.max_scaffolding[x][y],map[i][j]-abs(x-i)-abs(y-j))
                                u[x][y]+=(map[i][j]-(abs(x-i)+abs(y-j)))/map[i][j]
                                if x!=i or y!=j:
                                    self.useful_tree.add_node((x,y, map[i][j] - abs(x-i) - abs(y-j), 1))
                                    self.useful_tree.add_edge((i,j, map[i][j], 1),(x,y, map[i][j] - abs(x-i) - abs(y-j), 1))
        
        for node in self.useful_tree.nodes():
            if map[node[0]][node[1]]>0:
                successors = list(self.useful_tree.successors(node)) 
                if len(successors)>0:
                    for i in range(len(successors)):
                        if map[node[0]][node[1]] - successors[i][2]>1:
                            self.useful_tree.remove_edge(node, successors[i])
                        for j in range(len(successors)):
                            if successors[i][2]>successors[j][2]:
                                if abs(successors[i][0]-successors[j][0])+abs(successors[i][1]-successors[j][1])==1:
                                    self.useful_tree.add_edge(successors[i], successors[j])
        # self.visualize_useful_tree(self.useful_tree)
        return u
    
    def visualize_useful_tree(self, tree):
        # grid layout
        pos = nx.shell_layout(tree)

        # draw with data of nodes
        nx.draw_networkx_nodes(tree, pos, node_size=70)
        nx.draw_networkx_edges(tree, pos, edgelist=tree.edges(), width=2)
        nx.draw_networkx_labels(tree, pos, font_size=20, font_family='sans-serif')
        plt.axis('off')
        plt.show()

    def update_node_tree(self, node_map, reachability):
        tree = nx.DiGraph()
        for node in self.useful_tree.nodes():
            if (self.structure[node[0]][node[1]]>0 and node_map[node[0]][node[1]]<self.structure[node[0]][node[1]]):
                tree.add_node(node)

        for node in self.useful_tree.nodes():
            if node in tree.nodes():
                for successor in self.useful_tree.successors(node):
                    if successor in tree.nodes():
                        tree.add_edge(node, successor)
                        continue
                    elif node_map[successor[0]][successor[1]]<self.max_scaffolding[node[0]][node[1]]\
                        and node_map[successor[0]][successor[1]]>0\
                        and self.structure[successor[0]][successor[1]]==0:
                        tree.add_node(successor)
                        tree.add_edge(node, successor)
                    # (node_map[node[0]][node[1]]>0 and self.structure[node[0]][node[1]]==0 and node_map[node[0]][node[1]]<self.max_scaffolding[node[0]][node[1]])
                # for successor in self.useful_tree.successors(node):
                if len(list(tree.successors(node)))==0 and reachability[node[0]][node[1]]==0:
                    for successor in self.useful_tree.successors(node):
                        if reachability[successor[0]][successor[1]]==0:
                            tree.add_node(successor)
                            tree.add_edge(node, successor)   
        
        # self.visualize_useful_tree(self.useful_tree)
        return tree

    def reweight_edges(self):
        u = self.usefulness_matrix(self.structure)
        self.usefulness = u
        for e in self.H.edges():
            nmr = 1 + abs(self.structure[e[0][0]][e[0][1]] - self.structure[e[1][0]][e[1][1]])
            dnr = 1 + u[e[0][0]][e[0][1]] + u[e[1][0]][e[1][1]]
            temp_var = nmr / (dnr)
            self.H[e[0]][e[1]]['weight'] = temp_var

    def get_min_spanning_tree(self):
        G = nx.generators.lattice.grid_2d_graph(len(self.structure), len(self.structure[0]))
        self.H = nx.generators.lattice.grid_2d_graph(len(self.structure), len(self.structure[0]))    
        positions2 = {node: node for node in self.H.nodes()}
        positions2[(len(self.structure),len(self.structure[0]))]=(len(self.structure),len(self.structure[0]))

        self.reweight_edges()

        # add the source node
        for n in G.nodes():
            if n[0] == 0 or n[1] == 0 or n[0] == len(self.structure) - 1 or n[1] == len(self.structure[0]) - 1:
                self.H.add_edge((len(self.structure),len(self.structure[0])), n, weight=0)

        mst2 = nx.minimum_spanning_edges(self.H, algorithm='prim', data=False)
        edgelists2 = list(mst2)
        # edgelist_sorted = sorted(edgelists2)
        # plt.figure(figsize=(2*len(self.structure[0]),2*len(self.structure[0])))
        # nx.draw_networkx(self.H, positions2, width=2, node_size=15, edgelist=edgelists2)
        # plt.show()
        return edgelists2

def main():
    [goal] = np.array(GOAL_MAPS_12)
    mst = MinSpanningTree(goal)
    # print(mst.edges)

if __name__ == '__main__':
    main()