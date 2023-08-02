import min_spanning_tree
from goal import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb

class Lists():
    def __init__(self, struct):
        self.structure = struct

        mst = min_spanning_tree.MinSpanningTree(struct)
        self.edges = mst.edges

        self.tree = nx.DiGraph()
        root = (len(self.structure), len(self.structure[0]))
        self.tree.add_node(root)
        self.add_directed_edges(root)

        self.all_lists = dict()
        for node in self.tree:
            self.all_lists[node] = [0]

        self.construct_list(list(self.all_lists)[0])
        print(self.all_lists)
        
    def add_directed_edges(self, node):
        for e in self.edges:
            if e[0] == node:
                self.tree.add_edge(e[0], e[1])
                # print("added edge: ", e[0], " -> ", e[1])
                self.add_directed_edges(e[1])

    # def build_all_lists(self):

    def construct_list(self, N):
        # if N is a leaf node, add to list and return
        if len(list(self.tree.neighbors(N))) == 1:
            self.all_lists[N].append(self.structure[N[0]][N[1]])
            print("leaf node: ", N, " -> ", self.all_lists[N])
            return
        
        # call recursively for all N's children
        len_ = 0
        for child in self.tree.successors(N):
                print("child: ", child)
                self.construct_list(child)
                len_ = max(len_, len(self.all_lists[child]))
        # construct the i-th element of N's list
        for i in range(1, len_+1):
            if i%2 == 1:
                g_max = 0
                for child in self.tree.successors(N):
                    try:
                        g_max = max(g_max, self.all_lists[child][i])
                    except:
                        pass
                self.all_lists[N].append(max(self.all_lists[N][i-1], g_max))

            else:
                g_min = 255
                for child in self.tree.successors(N):
                    try:
                        g_min = min(g_min, self.all_lists[child][i])
                    except:
                        pass
                self.all_lists[N].append(min(self.all_lists[N][i-1], g_min))

        # construct the last element of N's list
        if N != list(self.tree)[0]:
            if len_%2 == 1 and list(self.all_lists[N])[-1]<=self.structure[N[0]][N[1]]:
                self.all_lists[N][len_] = self.structure[N[0]][N[1]]

            elif len_%2 == 1 and list(self.all_lists[N])[-1]>self.structure[N[0]][N[1]]:
                self.all_lists[N].append(self.structure[N[0]][N[1]])

            elif len_%2 == 0 and list(self.all_lists[N])[-1]>=self.structure[N[0]][N[1]]:
                self.all_lists[N][len_] = self.structure[N[0]][N[1]]

            else:
                self.all_lists[N].append(self.structure[N[0]][N[1]])
        else:
            self.all_lists[N].append(0)

                    
                

def main():
    [goal] = np.array(GOAL_MAPS_12)
    list_ = Lists(goal)

if __name__ == '__main__':
    main()
        