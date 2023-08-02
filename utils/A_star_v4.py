import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import min_spanning_tree 
from goal import *
import heapq
from itertools import count
import reachability as rm
import time


class HighLevelPlanner:
    def __init__(self) -> None:
        self.goal_state = GOAL_MAPS_12[0]
        self.mst = min_spanning_tree.MinSpanningTree(self.goal_state)
        self.edges = self.mst.edges
        self.reachability, _ = rm.get_reachability(np.zeros((len(self.goal_state), len(self.goal_state[0]))))
        # self.dynamic_usefulness = self.compute_dynamic_usefulness(self.goal_state)
        # self.plan = self.get_plan()
        self.usefulness = self.mst.usefulness

    def compute_dynamic_usefulness(self, node_map):
        # covert node map to integer matrix
        node_map = node_map.astype(int)
        u = self.mst.dyn_usefulness_matrix(node_map)
                
        

    def compute_heuristics(self, node_map):
        matrix = node_map - (self.goal_state)
        h_value = np.sum(np.abs(matrix))
        return h_value

    def get_plan(self, node):
        plan = []
        while node['parent'] is not None:
            plan.append(node['action'])
            node = node['parent']
        plan.reverse()
        print(f'Number of actions: {len(plan)-1}')
        print(f'Time taken: {time.time() - self.start_time}')
        return plan
        # print("plan", plan)
    
    def check_validity(self, node_map, i, j, k):
        nbrs = []
        for edge in self.edges: 
            if edge[0] == (i,j) or edge[1] == (i,j):
                nbrs.append(edge[0] if edge[0] != (i,j) else edge[1])
        if len(nbrs) == 0:
            return False
        else:
            for nbr in nbrs:
                if node_map[nbr]==k-1:
                    #and node_map==k
                    return True
                
    def get_actions(self, node_map):
        actions = []
        max_height = 5
        for i in range(node_map.shape[0]):
            for j in range(node_map.shape[1]):
                if self.reachability[i,j] == 0:
                    if node_map[i,j] < max_height and self.usefulness[i,j] > 0:
                        if self.check_validity(node_map, i, j, node_map[i,j]+1):
                            new_map = np.copy(node_map)
                            new_map[i,j] += 1
                            actions.append({'action': (i, j, node_map[i, j], 1), 'cost': 1, 'state': new_map})
                    if node_map[i,j] > 0 and node_map[i,j] > self.goal_state[i][j]:
                        if self.check_validity(node_map, i, j, node_map[i,j]):
                            new_map = np.copy(node_map)
                            new_map[i,j] -= 1
                            actions.append({'action': (i, j, node_map[i, j], -1), 'cost': 1, 'state': new_map})
        return actions

    def a_star(self, start_state, goal_state):
        self.start_time = time.time()
        open_list = []
        closed_list = dict()
        tiebreaker = count(step=1)
        state = start_state

        root = {'action': (-1,-1,-1,0), 'g_val': 0, 'h_val': self.compute_heuristics(start_state), 'parent': None}
        heapq.heappush(open_list, (root['g_val'] + root['h_val'], root['h_val'], next(tiebreaker),root))
        closed_list[(root['action'])] = root
        gen = expand = invalid = dup = dup2 = 0

        while len(open_list) > 0:
            curr = heapq.heappop(open_list)[3]
            expand += 1
            state[curr['action'][0]][curr['action'][1]] += curr['action'][2]
            self.reachability, _ = rm.get_reachability(state)
            self.dynamic_usefulness = self.compute_dynamic_usefulness(state)
            
            if np.array_equal(state, goal_state):
                print("Found goal_state")
                print(f'Generated: {gen}, Expanded: {expand}, Invalid: {invalid}, Duplicate: {dup}, Duplicate2: {dup2}')
                
                return self.get_plan(curr)
            
            for new_action in self.get_actions(state):
                child = {'action': new_action['action'], 'g_val': curr['g_val'] + new_action['cost'], 'h_val': self.compute_heuristics(new_action['state']), 'parent': curr}
                if child['action'] in closed_list:
                    existing_node = closed_list[child['action']]
                    if existing_node['g_val'] <= child['g_val']:
                        continue
                gen += 1
                heapq.heappush(open_list, (child['g_val'] + child['h_val'], child['h_val'], next(tiebreaker),child))
                closed_list[child['action']] = child

def main():
    hlp = HighLevelPlanner()
    size = len(hlp.goal_state)
    hlp.a_star(np.zeros((size, size)), hlp.goal_state)

if __name__ == '__main__':
    main()
