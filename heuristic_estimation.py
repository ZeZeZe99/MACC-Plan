import numpy as np
import networkx as nx
import reachability as rm

def estimate_h_value(max_scaff, goal):
    print(max_scaff, goal)
    goal_reachable = rm.get_reachability(goal)
    true_scaff = np.zeros(max_scaff.shape)
    while True:
        if np.abs(goal - goal_reachable).sum():
            return print(goal- goal_reachable)
        else:
            
            pass

