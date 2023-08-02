import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import min_spanning_tree 
from goal import *
import heapq
from itertools import count
from reachability import PathFinder
import time
# import A_star as astar_planner
from A_star import heuristic
import config
import lego

arg = config.get_parser()
arg = arg.parse_args()
env = lego.GridWorld(arg)

class HighLevelPlanner:
    def __init__(self) -> None:
        self.goal_state = GOAL_MAPS_12[0]
        self.mst = min_spanning_tree.MinSpanningTree(self.goal_state)
        self.edges = self.mst.edges
        self.pf = PathFinder(np.zeros((len(self.goal_state), len(self.goal_state[0]))))
        self.reachability = self.pf.reachability
        self.usefulness = self.mst.usefulness

    def compute_heuristics(self, node_map):
        # matrix = node_map - (self.goal_state)
        # h_value = np.sum(np.abs(matrix))
        env.height = node_map
        h_value = heuristic(env, env.height, mode=1)
        return h_value

    def get_plan(self, node):
        plan = []
        action_tree = []
        while node['parent'] is not None:
            plan.append(node['state'])
            action_tree.append((node['tree'], node['state'], node['parent']['state']))
            node = node['parent']
        plan.reverse()
        print(f'Number of actions: {len(plan)-1}')
        print(f'Time taken: {time.time() - self.start_time}')

        ADG = self.build_action_dep_tree(action_tree)
        return plan
    
    def build_action_dep_tree(self, action_tree):
        ADG = nx.DiGraph()
        for node in action_tree:
            x = np.nonzero(node[1]-node[2])[0][0]
            y = np.nonzero(node[1]-node[2])[1][0]
            if (x, y, int(node[2][x, y]), int(node[1][x, y]-node[2][x, y])) not in ADG.nodes():
                ADG.add_node((x, y, int(node[2][x, y]), int(node[1][x, y]-node[2][x, y])))
            if node[0] is not None:
                for n in node[0].nodes():
                    if n[0] == x and n[1] == y:
                        p = list(node[0].predecessors(n))
                        for p_ in p:
                            for k in node[0].nodes():
                                if k[0] == p_[0] and k[1] == p_[1]:
                                    ADG.add_edge((x, y, int(node[2][x, y]), int(node[1][x, y]-node[2][x, y])), k)
        # visualize the graph
        # pos = nx.shell_layout(ADG)
        # nx.draw(ADG, pos, with_labels=True)
        # plt.show()
        # for parent in ADG.nodes():
        #     for child in ADG.nodes():
        #         if parent!=child:
        #             # is child an operation on parent location?
        #             if parent[0] == child[0] and parent[1] == child[1] and (parent[2]>child[2] and parent[3]==child[3]):
        #                 ADG.add_edge(parent, child)
                    
            

    def check_validity(self, node_map, i, j, k):
        nbrs_ = {(i-1,j), (i+1,j), (i,j-1), (i,j+1)}
        for nbr in nbrs_:
            # if nbr in x for x in self.edges:
                if node_map[nbr] == k-1:
                    return True
        # nbrs = []
        # for edge in self.edges: 
        #     if edge[0] == (i,j) or edge[1] == (i,j):
        #         nbrs.append(edge[0] if edge[0] != (i,j) else edge[1])
        # if len(nbrs) == 0:
        #     return False
        # else:
        #     for nbr in nbrs:
        #         if node_map[nbr]==k-1:
        #             #and node_map==k
        #             return True

    def get_tree(self, node_map):
        return self.mst.update_node_tree(node_map, self.reachability)

    def get_actions(self, node_map, tree):
        actions = []
        if tree is None:
            for i in range(len(node_map)):
                for j in range(len(node_map[0])):
                    if self.goal_state[i][j]>0:
                        new_map = np.copy(node_map)
                        new_map[i,j] += 1
                        actions.append({'state': new_map})
        else:
            for i in range(1, node_map.shape[0]-1):
                for j in range(1, node_map.shape[1]-1):
                    if any((node[0] == i and node[1] == j) for node in tree):
                    # and node_map[i,j] < self.mst.max_scaffolding[i, j]:
                        if self.check_validity(node_map, i, j, node_map[i,j]+1):
                            new_map = np.copy(node_map)
                            new_map[i,j] += 1
                            # print(i, j, node_map[i, j], 1)
                            actions.append({'state': new_map})

                    elif node_map[i,j] > self.goal_state[i][j]:
                        if self.check_validity(node_map, i, j, node_map[i,j]):
                            new_map = np.copy(node_map)
                            new_map[i,j] -= 1
                            # print(i, j, node_map[i, j], -1)
                            actions.append({'state': new_map})
    # for i in range(len(actions)):
        #     for j in range(i, len(actions)):
        #         if np.array_equal(actions[i]['state'], actions[j]['state']):
        #             actions.pop(j)
        #             break
        return actions

    def a_star(self, start_state, goal_state):
        self.start_time = time.time()
        open_list = []
        closed_list = dict()
        tiebreaker = count(step=-1)
        root = {'state': start_state, 'g_val': 0, 'h_val': self.compute_heuristics(start_state), 'parent': None, 'tree': None, 'reachability': self.reachability}
        heapq.heappush(open_list, (root['g_val'] + root['h_val'], root['h_val'], next(tiebreaker),root))
        closed_list[(root['state']).tobytes()] = root
        gen = expand = invalid = dup = dup2 = 0
        cost = 1
        symm = dict()
        while len(open_list) > 0:
            curr = heapq.heappop(open_list)[3]
            expand += 1
            # self.reachability, _ = self.pf.get_reachability(curr['state'], self.reachability)
            self
            cost = curr['g_val']
            if np.array_equal(curr['state'], goal_state):
                print("Found goal_state")
                print(f'Generated: {gen}, Expanded: {expand}, Invalid: {invalid}, Duplicate: {dup}, Duplicate2: {dup2}')      
                return self.get_plan(curr)
            
            for action in self.get_actions(curr['state'], curr['tree']):
                child = {'state': action['state'], 'g_val': curr['g_val'] + 1, 'h_val': self.compute_heuristics(action['state']), 'parent': curr, 'tree': self.get_tree(action['state']), 'reachability': self.pf.get_reachability(curr['state'], curr['reachability'], (np.nonzero(curr['state']-action['state'])))}  
                
                #symmetry
                
                # usefulness_sum = np.zeros((len(self.usefulness), len(self.usefulness[0])))
                usefulness_sum = 0
                for i in range(len(start_state)):
                    for j in range(len(start_state[0])):
                        if child['state'][i, j] > 0:
                            for k in range(int(child['state'][i, j])):
                                usefulness_sum += self.usefulness[i, j]
                
                if any(((child['h_val']==symm[k][0]) and (usefulness_sum==symm[k][1])) for k in symm.keys()):
                    # print("symmetry duplicate")
                    dup += 1
                    continue

                # symm[child['state'].tobytes()] = (child['h_val'], usefulness_sum)
                
                if child['state'].tobytes() in closed_list:
                    existing_node = closed_list[child['state'].tobytes()]
                    if existing_node['g_val'] <= child['g_val']:
                        continue
                gen += 1

                heapq.heappush(open_list, (child['g_val'] + child['h_val'], child['h_val'], next(tiebreaker),child))
                closed_list[child['state'].tobytes()] = child

def main():
    hlp = HighLevelPlanner()
    size = len(hlp.goal_state)
    hlp.a_star(np.zeros((size, size)), hlp.goal_state)

if __name__ == '__main__':
    main()