import heapq
import pdb

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def is_valid_motion(old_loc, new_loc):
    # Check if two agents are in the same location (vertex collision)
    if len(set(new_loc)) != len(new_loc):
        return False
    # Check edge collision
    for i in range(len(new_loc)):
        for j in range(len(old_loc)):
            if i == j:
                continue
            if new_loc[i] == old_loc[j] and new_loc[j] == old_loc[i]:
                return False
    return True

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst

def compute_heuristics(my_map, adjacency_matrix, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            # skip if edge does not exist
            loc_idx = loc[0]*8 + loc[1]
            child_loc_idx = child_loc[0]*8 + child_loc[1]
            if adjacency_matrix[loc_idx, child_loc_idx] == 0:
                # print("skipping no edge", loc, child_loc)
                continue
            # skip if child_loc is already occupied
            if my_map[child_loc[0]][child_loc[1]]:
                # print("skipping occupied")
                continue
            # add child to the open_list
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
                    # print("child added to open 1", child)
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))
                # print("child added to open 2", child)
    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def build_constraint_table(path, agent_id, my_map, constraints):
    # lookupTable = dict()
    # max_timestep = 0
    # print("constraints", constraints)
    # print("path", path)
    # print("agent", agent_id)
    if len(constraints[agent_id]) == 0:
        for t in range(len(path)):
            constraints[agent_id].append((path[t], t))
            my_map[t][path[t][0]][path[t][1]] = 1
    else:
        for t in range(len(constraints[agent_id]), len(constraints[agent_id])+len(path)):
            constraints[agent_id].append((path[t-len(constraints[agent_id])], t))
            my_map[t][path[t-len(constraints[agent_id])][0]][path[t-len(constraints[agent_id])][1]] = 1
    
    
    ##############################
    # Task 1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    ##############################
    return constraints, my_map

def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    ##############################

    return False

def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location

def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        # print('loc: {}, timestep: {}'.format(curr['loc'], curr['timestep']))
        curr = curr['parent']
    path.reverse()
    return path

def push_node(open_list, node):
    # print(node['g_val'], node['h_val'], node['loc'])
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))

def push_node_js(open_list, node):
    # print(node['g_val'], node['h_val'], node['loc'])
    # f_val  = [node['g_val'][i] + node['h_val'][i]] for i in range(len(node['g_val']))
    f_val = [node['g_val'][i] + node['h_val'][i] for i in range(len(node['g_val']))]
    heapq.heappush(open_list, (f_val, node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

def no_obstacles(map, locs):
    for loc in locs:
        if map[loc[0]][loc[1]]:
            # print("obstacle at ", loc)
            return False
    # print("no obstacles found", locs)
    return True

def is_boundary_cell(i, j, height, width):
    if i == 0 or i == height-1 or j == 0 or j == width-1:
        return True
    else:
        return False


def a_star(total_map, start_loc, goal_loc, h_values, agent_id, constraints, adjacency_matrix):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    ##############################
    # Task 1.2: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    
    
    open_list = []
    closed_list = dict()
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'])] = root
    
    max_timestep = len(constraints[agent_id])
    print("max timestep", max_timestep, "of agent", agent_id)
    timestep = max_timestep
    # pdb.set_trace()
    while len(open_list) > 0:
        my_map = total_map[timestep]
        curr = pop_node(open_list)
        timestep+=1
        #############################
        # Task 2.3: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            print("found goal", get_path(curr))
            return get_path(curr)
        #############################
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            loc_idx = curr['loc'][0]*8 + curr['loc'][1]
            child_loc_idx = child_loc[0]*8 + child_loc[1]
            print("child_loc", child_loc, "curr_loc", curr['loc'], "goal_loc", goal_loc)
            if child_loc[0]<0 or child_loc[0]>=len(my_map) or child_loc[1]<0 or child_loc[1]>=len(my_map[0]):
                # pdb.set_trace()
                continue
            elif my_map[child_loc[0]][child_loc[1]]:
                # pdb.set_trace()
                continue
            elif adjacency_matrix[loc_idx, child_loc_idx] == 0:
                # pdb.set_trace()
                # print("skipping no edge", curr['loc'], child_loc)
                continue
            try: 
                h_values[child_loc]
                continue
            except:
                continue
            print("no obstacle, within boundary and edge exists")
            
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr}
            if (child['loc']) in closed_list:
                existing_node = closed_list[(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'])] = child
                push_node(open_list, child)
        
    ##############################
    return None  # Failed to find solutions
