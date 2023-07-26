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

    def reweight_edges(self, H):
        
        u = np.zeros((len(self.structure), len(self.structure[0])))
        for i in range(0, len(self.structure)):
            for j in range(0, len(self.structure[0])):
                if self.structure[i][j]>0:
                    for x in range(i - self.structure[i][j]-1, i+ self.structure[i][j]):
                        for y in range(j - self.structure[i][j]-1, j+ self.structure[i][j]):
                            if (abs(x-i)+abs(y-j))<self.structure[i][j]:
                                u[x][y]+=(self.structure[i][j]-(abs(x-i)+abs(y-j)))/self.structure[i][j]
        self.usefulness = u
        print(self.usefulness)
        for e in H.edges():
            nmr = 1 + abs(self.structure[e[0][0]][e[0][1]] - self.structure[e[1][0]][e[1][1]])
            dnr = 1 + u[e[0][0]][e[0][1]] + u[e[1][0]][e[1][1]]
            temp_var = nmr / (dnr)
            H[e[0]][e[1]]['weight'] = temp_var
        return H        

    def get_min_spanning_tree(self):
        G = nx.generators.lattice.grid_2d_graph(len(self.structure), len(self.structure[0]))
        H = nx.generators.lattice.grid_2d_graph(len(self.structure), len(self.structure[0]))    
        positions2 = {node: node for node in H.nodes()}
        positions2[(len(self.structure),len(self.structure[0]))]=(len(self.structure),len(self.structure[0]))

        H = self.reweight_edges(H)
        for n in G.nodes():
            if n[0] == 0 or n[1] == 0 or n[0] == len(self.structure) - 1 or n[1] == len(self.structure[0]) - 1:
                H.add_edge((len(self.structure),len(self.structure[0])), n, weight=0)

        mst2 = nx.minimum_spanning_edges(H, algorithm='kruskal', data=False)
        edgelists2 = list(mst2)
        edgelist_sorted = sorted(edgelists2)
        plt.figure(figsize=(2*len(self.structure[0]),2*len(self.structure[0])))
        nx.draw_networkx(H, positions2, width=2, node_size=15, edgelist=edgelists2)
        plt.show()
        return edgelist_sorted

def main():
    [goal] = np.array(GOAL_MAPS_10)
    mst = MinSpanningTree(goal)
    print(mst.edges)

if __name__ == '__main__':
    main()