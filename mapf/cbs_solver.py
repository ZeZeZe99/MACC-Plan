import time as timer
import heapq
import random
import numpy as np
import dep_graph as dg
import pdb
import networkx as nx
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost, get_location

def detect_collision(path1, path2):
    return None

def detect_collisions(paths):
    collisions = []
    return collisions

def resolve_collision(collision):
    constraints = []
    return constraints

def get_obstacle_map(my_map):
    graph = nx.Graph()
    for i in range(len(my_map)):
        for j in range(len(my_map[0])):
            if i > 0:
                if abs(my_map[i][j] - my_map[i-1][j]) <= 1:
                    graph.add_edge((i,j), (i-1,j))
            if i < len(my_map) - 1:
                if abs(my_map[i][j] - my_map[i+1][j]) <= 1:
                    graph.add_edge((i,j), (i+1,j))
            if j > 0:
                if abs(my_map[i][j] - my_map[i][j-1]) <= 1:
                    graph.add_edge((i,j), (i,j-1))
            if j < len(my_map[0]) - 1:
                if abs(my_map[i][j] - my_map[i][j+1]) <= 1:
                    graph.add_edge((i,j), (i,j+1))
    obstacle_map = nx.adjacency_matrix(graph)
    return obstacle_map


class CBSSolver():
    def __init__(self, node_list, agents, height_map, iteration):
        self.iteration = iteration
        self.my_map = height_map
        self.obstacle_map_adjacency_matrix = get_obstacle_map(self.my_map[iteration])
        self.starts = agents.locations
        self.goals = node_list
        self.num_of_agents = len(self.goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.open_list = []
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(self.my_map, goal))
        #self.find_solution()

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        self.start_time = timer.time()
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                            i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        
        raise BaseException('No solutions')
    