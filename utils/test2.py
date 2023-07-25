import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb
import heapq
import dep_graph as dg
from itertools import count

# Data: Action Dependency Graph G
# Result: List of tasks allocated to each agent
# TaskList = Topological Sort of G
# Initialize t ← 0, and all agents as “free” at all t
# while TaskList not empty do
# if any agent is free at time t then
# Pop task from TaskList
# For each free agent compute TaskTime
# Allocate task to agent A with min. TaskTime
# Set A not free for next TaskTime steps
# end
# t ← t +1
# end

env_size = 6
class Agent_Task_Allocation:
    def __init__(self, agent_id):
        self.free = True
        self.time = 0
        self.agent_id = agent_id
        self.location = (-1, -1, -1)
        self.tasks = []
        self.task_parents = []
        self.min_task_start_time = []
        self.min_task_end_time = []

    def add_task(self, task, parents, map):
        self.free = False
        self.task_parents.append((parents))
        if len(self.min_task_end_time) > 0:
            self.min_task_start_time.append(self.min_task_end_time[-1])
        else: 
            self.min_task_start_time.append(0)
        self.min_task_end_time.append(self.task_time(task, map))
        self.tasks.append(task)

    def task_time(self, new_task, map):
        global env_size
        if len(self.tasks)>0:
            self.find_targets(new_task, self.tasks[-1], map)
        else:
            self.find_targets(new_task, None, map)
        return 0
    
    def find_targets(self, task, prev_task, map):
        global env_size
        if prev_task == None:
            goal_loc = task[0]
            agent_loc = self.location
            path = self.move_to_target(agent_loc, goal_loc, map)
            print("prev task none")
            self.update_map(path, map)
            self.location = path[-1]
        elif prev_task[1] != task[1]: 
            goal_loc = task[0]
            agent_loc = self.location
            path = self.move_to_target(agent_loc, goal_loc, map)
            print("prev task Carry/Not Carry not equals current task Carry/Not Carry")
            self.update_map(path, map)
            self.location = path[-1]
        else: 
            goal_loc = (-1, -1, -1)
            agent_loc = self.location
            path1 = self.move_to_target(agent_loc, goal_loc, map)
            self.update_map(path, map)
            self.location = path1[-1]
            goal_loc = task[0]
            agent_loc = self.location
            path2 = self.move_to_target(agent_loc, goal_loc, map)
            self.update_map(path, map)
            self.location = path2[-1]

    def update_map(self, path, map):
        global env_size
        timestep = self.time
        print("path", path)
        for loc in path:
            map[self.time+timestep][loc[0]][loc[1]][loc[2]] = self.agent_id
            timestep += 1

    def move_to_target(self, agent_loc, goal_loc, map):
        global env_size
        # open_list = []
        # closed_list = dict()
        # h_start = abs(agent_loc[0]-goal_loc[0])+abs(agent_loc[1]-goal_loc[1])+abs(agent_loc[2]-goal_loc[2])
        # root = {'loc': agent_loc, 'g': 0, 'h': h_start, 'parent': None}
        # open_list.append(root)
        # closed_list[(root['loc'])] = root
        # while(len(open_list)>0):
        #     current = open_list.pop(0)
        #     print("open", len(open_list))
        #     print("closed", len(closed_list))
        #     if current['loc'] == goal_loc:
        #         return self.get_path(current)
        #     for dir in range(4):
        #         child_loc = self.move(current['loc'], dir, goal_loc)
        #         if child_loc[0]<0 or child_loc[0]>=env_size or child_loc[1]<0 or child_loc[1]>=env_size or child_loc[2]<0 or child_loc[2]>=3:
        #             continue
        #         if map[self.time][child_loc[0]][child_loc[1]][child_loc[2]]: #(0 or -1 are traversable, all other numbers are obstacles)
        #             continue
        #         child = {'loc': child_loc, 
        #                  'g': current['g']+1, 
        #                  'h': abs(child_loc[0]-goal_loc[0])+abs(child_loc[1]-goal_loc[1])+abs(child_loc[2]-goal_loc[2]),
        #                  'parent': current}
        #         if (child['loc']) in closed_list:
        #             existing_node = closed_list[(child['loc'])]
        #             if child['g']+child['h'] < existing_node['g']+existing_node['h']:
        #                 closed_list[(child['loc'])] = child
        #                 open_list.append(child)
        #         else:
        #             closed_list[(child['loc'])] = child
        #             open_list.append(child)
        open_list = []
        closed_list = dict()
        tiebreaker = count(step=-1)

    def move(self, loc, dir, goal_loc):
        global env_size
        if loc!=(-1,-1,-1):
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
            return loc[0] + directions[dir][0], loc[1] + directions[dir][1], loc[2]
        else:
            # return boundary location closest to the goal
            if goal_loc[0] < env_size/2:
                x = 0
            else:
                x = env_size-1
            if goal_loc[1] < env_size/2:
                y = 0
            else:
                y = env_size-1
            return x, y, 0

    def get_path(self, goal_node):
        path = []
        curr = goal_node
        while(curr!=None):
            path.append(curr['loc'])
            curr = curr['parent']
        path.reverse()
        return path
         
def main(max_agents = 6):
    Action_dep_graph = adg.main()
    ADG_copy = Action_dep_graph.copy()
    agents = []
    map = np.zeros((60, 6, 6, 3))
    for i in range(max_agents):
        agents.append(Agent_Task_Allocation(i+1))
    while(Action_dep_graph.number_of_nodes() > 0):
        for agent in agents:
            if agent.free:
                for node in Action_dep_graph:
                    if len(Action_dep_graph.succ[node]) == 0:
                        agent.add_task(node, ADG_copy.succ[node], map)
                        Action_dep_graph.remove_node(node)
                        break
        for agent in agents:
            agent.time += 1
            if len(agent.min_task_end_time) > 0:
                if agent.time >= agent.min_task_end_time[-1]:
                    agent.free = True
    return agents, ADG_copy

if __name__ == '__main__':
    main()