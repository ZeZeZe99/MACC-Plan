import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import min_spanning_tree 
from goal import *
import heapq
from itertools import count

class HighLevelPlanner:
    def __init__(self) -> None:
        self.goal_state = GOAL_MAPS_10[0]
        mst = min_spanning_tree.MinSpanningTree(self.goal_state)
        self.edges = mst.edges
        self.usefulness = mst.usefulness
        # self.plan = self.get_plan()

    def compute_heuristics(self, node_map):
        matrix = node_map - (self.goal_state)
        h_value = np.sum(np.abs(matrix))
        return h_value

    def get_plan(self, node):
        plan = []
        while node['parent'] is not None:
            plan.append(node['state'])
            node = node['parent']
        plan.reverse()
        print("plan", plan)
        # action_sequence = self.build_ADG(plan)
        # return action_sequence
    
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
                if node_map[i,j] < max_height and self.usefulness[i,j] > 0:
                    # add actions
                    # add only inside the shadow region of the goal
                    if self.check_validity(node_map, i, j, node_map[i,j]+1):
                        new_map = np.copy(node_map)
                        new_map[i,j] += 1
                        actions.append({'state': new_map, 'cost': 1})
                if node_map[i,j] > 0 and node_map[i,j] > self.goal_state[i][j]:
                    if self.check_validity(node_map, i, j, node_map[i,j]):
                        new_map = np.copy(node_map)
                        new_map[i,j] -= 1
                        actions.append({'state': new_map, 'cost': 1})
        return actions

    def a_star(self, start_state, goal_state):
        open_list = []
        closed_list = dict()
        tiebreaker = count(step=-1)
        root = {'state': start_state, 'g_val': 0, 'h_val': self.compute_heuristics(start_state), 'parent': None}
        heapq.heappush(open_list, (root['g_val'] + root['h_val'], root['h_val'], next(tiebreaker),root))
        closed_list[(root['state']).tobytes()] = root
        while len(open_list) > 0:
            curr = heapq.heappop(open_list)[3]
            print("curr", curr['state'])
            # self.reachability, self.reachable_nbrs = rm.get_reachability(curr['state'])
            if np.array_equal(curr['state'], goal_state):
                print("Found goal_state")
                return self.get_plan(curr)
            for action in self.get_actions(curr['state']):
                child = {'state': action['state'], 'g_val': curr['g_val'] + action['cost'], 'h_val': self.compute_heuristics(action['state']), 'parent': curr}
                if child['state'].tobytes() in closed_list:
                    existing_node = closed_list[child['state'].tobytes()]
                    if existing_node['g_val'] <= child['g_val']:
                        continue
                heapq.heappush(open_list, (child['g_val'] + child['h_val'], child['h_val'], next(tiebreaker),child))
                closed_list[child['state'].tobytes()] = child

def main():
    hlp = HighLevelPlanner()
    size = len(hlp.goal_state)
    hlp.a_star(np.zeros((size, size)), hlp.goal_state)

if __name__ == '__main__':
    main()
