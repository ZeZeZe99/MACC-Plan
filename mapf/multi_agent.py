import sys
import networkx as nx
import numpy as np
import dep_graph as dg
import heapq
from itertools import count
import mapf.pp_solver as pp
import mapf.cbs_solver as cbs
import time as timer

class Agents():
    def __init__(self):
        self.num_agents = 6
        self.free_agents = dict()
        self.time = 0
        self.locations = dict()
        for i in range(6):
            # i is agent id
            self.free_agents[i] = True
            self.locations[i] = (-1, -1, -1)

def task_allocation():
    graph = dg.main()
    graph.adg.remove_node((-1, -1, -1, 0))
    # skip data[0] as empty node
    avg_makespan = 40
    height_map = np.zeros((avg_makespan, 8, 8)) 
    agents = Agents()
    iteration = 0
    num_agents = 3
    constraints = dict()
    for i in range(num_agents):
        constraints[i] = []
    while(len(graph.adg.nodes())>0):
        node_list = []
        for node in graph.adg.nodes():
            if len(graph.adg.pred[node])==0:
                node_list.append(node)
        
        pp.PPSolver(node_list, agents, height_map, iteration, constraints)
        # cbs.CBSSolver(node_list, agents, height_map, iteration)
        iteration += 1
        for node in node_list:
            graph.adg.remove_node(node)

if __name__ == '__main__':
    task_allocation()