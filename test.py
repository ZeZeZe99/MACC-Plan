import sys
import networkx as nx
import numpy as np
import dep_graph as g
import heapq
from itertools import count

class Agent:
    def __init__(self, agent_id):
        self.free = True
        self.time = 0
        self.agent_id = agent_id
        self.location = (-1, -1, -1)
        self.tasks = []
        self.task_parents = []
        self.min_task_start_time = []
        self.min_task_end_time = []

    def add_task_to_agent(self):
        pass

    def calculate_task_time(self, new_task, map):
        pass

    def find_targets(self, new_task, prev_task, map):
        pass

    def calculate_h_value(self, loc, goal):
        h =0
        return h
    
    def get_actions(self, curr, goal, map):
        actions = []
        
        return actions
    
    def get_path_to_target(self, goal, map):
        global env_size
        open_list = []
        closed_list = dict()
        tiebreaker = count(step=-1)
        root = {'loc': self.location, 'g': 0, 'h': self.calculate_h_value(self.location, goal), 'parent': None}
        heapq.heappush(open_list, (root['g']+root['h'], root['h'], next(tiebreaker), root))
        closed_list[(root['loc'])] = root
        while len(open_list) > 0:
            curr = heapq.heappop(open_list)[3]
            if curr['loc'] == goal:
                return self.get_path(curr)
            for child in self.get_actions(curr, goal, map):
                if child['loc']:
                    pass

    def move_to_target(self, goal_loc, map):
        total_path = []
        while len(goal_loc)>0:
            goal = goal_loc.pop(0)
            path = self.get_path_to_target(goal, map)
            if len(path) > 0:
                total_path.extend(path)
        return total_path
    
def main():
    graph = g.main()

if __name__ == '__main__':
    main()
