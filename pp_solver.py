import time as timer
import heapq
import random
import numpy as np
import dep_graph as dg
import pdb
import networkx as nx
from single_agent_planner import compute_heuristics, a_star, build_constraint_table, is_boundary_cell

class PPSolver():
    def __init__(self, node_list, agents, height_map, timestep, constraints):
        self.timestep = timestep  
        self.my_map = height_map 
        self.starts = agents.locations
        self.goals = node_list
        self.num_of_goals = len(self.goals)
        self.adjacency_matrix = self.get_obstacle_map(self.my_map[timestep]) #adjaency matrix
        self.CPU_time = 0
        self.heuristics = []
        self.constraints = constraints
        self.find_solution()

    def get_obstacle_map(self, my_map):
        graph = nx.Graph()
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                if i > 0:
                    if abs(my_map[i][j] - my_map[i-1][j]) <= 1:
                        graph.add_edge((i,j), (i-1,j))
                        graph.add_edge((i-1,j), (i,j))
                if i < len(my_map) - 1:
                    if abs(my_map[i][j] - my_map[i+1][j]) <= 1:
                        graph.add_edge((i,j), (i+1,j))
                        graph.add_edge((i+1,j), (i,j))
                if j > 0:
                    if abs(my_map[i][j] - my_map[i][j-1]) <= 1:
                        graph.add_edge((i,j), (i,j-1))
                        graph.add_edge((i,j-1), (i,j))
                if j < len(my_map[0]) - 1:
                    if abs(my_map[i][j] - my_map[i][j+1]) <= 1:
                        graph.add_edge((i,j), (i,j+1))
                        graph.add_edge((i,j+1), (i,j))
        obstacle_map = np.zeros((len(my_map)*len(my_map[0]), len(my_map)*len(my_map[0])))
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                if i > 0:
                    if abs(my_map[i][j] - my_map[i-1][j]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, (i-1)*len(my_map[0])+j] = 1
                        obstacle_map[(i-1)*len(my_map[0])+j, i*len(my_map[0])+j] = 1
                if i < len(my_map) - 1:
                    if abs(my_map[i][j] - my_map[i+1][j]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, (i+1)*len(my_map[0])+j] = 1
                        obstacle_map[(i+1)*len(my_map[0])+j, i*len(my_map[0])+j] = 1
                if j > 0:
                    if abs(my_map[i][j] - my_map[i][j-1]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, i*len(my_map[0])+j-1] = 1
                        obstacle_map[i*len(my_map[0])+j-1, i*len(my_map[0])+j] = 1
                if j < len(my_map[0]) - 1:
                    if abs(my_map[i][j] - my_map[i][j+1]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, i*len(my_map[0])+j+1] = 1
                        obstacle_map[i*len(my_map[0])+j+1, i*len(my_map[0])+j] = 1
        return obstacle_map

    def compute_start_and_goal_loc(self, i, heuristic):
        goal_loc = (self.goals[i][0], self.goals[i][1])
        start_loc = (self.starts[i][0], self.starts[i][1])
        my_map = self.my_map[self.timestep]

        if start_loc == (-1, -1):
            min_h = float('inf')
            for i in range(len(my_map)):
                for j in range(len(my_map[0])):
                    if my_map[i][j] == 0 and is_boundary_cell(i, j, len(my_map), len(my_map[0])):
                        h = heuristic[(i,j)]
                        if h < min_h:
                            min_h = h
                            new = (i,j)
            start_loc = new
        
        return start_loc, goal_loc

    def update_my_map(self, result):
        for i in range(self.num_of_goals):
            try:
                path = result[i]
                build_constraint_table(path, i)
            except:
                continue
    
    def find_free_agent(self, constraints):
        return random.randint(0, 2)

    def find_solution(self):
        start_time = timer.time()
        result = dict()
        constraints = self.constraints
        timestep = 0
        
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics(self.my_map[timestep], self.adjacency_matrix, (self.goals[i][0], self.goals[i][0])))
            print("Num constraints {}".format(len(constraints)))
            start_loc, goal_loc = self.compute_start_and_goal_loc(i, self.heuristics[i])
            
            agent_id = self.find_free_agent(constraints)
            path = a_star(self.my_map, start_loc, goal_loc, self.heuristics[i],
                          agent_id, constraints, self.adjacency_matrix)
            if path is None:
                raise BaseException('No solutions')
            
            constraints, self.my_map = build_constraint_table(path, agent_id, self.my_map, constraints)
        
        self.CPU_time = timer.time() - start_time
        # print("Time to find Low Level Sol in roun", self.CPU_time)
        


if __name__ == '__main__':
    PPSolver()