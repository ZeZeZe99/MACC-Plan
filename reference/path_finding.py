import heapq
from collections import deque
from copy import deepcopy

a_star_mode = 1  # 0: Multi-step A*, 1: Multi-label A*


'''Goal processing'''
def process_goal(goals, height, carry_stats):
    """Process goal information"""
    # Add id to each goal
    for i, g in enumerate(goals):
        goals[i] = (g[0], g[1], g[2], g[3], i)

    info = dict()
    info['goals'] = goals
    # Mark start stage for each agent
    info['stage'] = dict()
    for i in range(len(goals)):
        add = goals[i][0]
        carry = carry_stats[i]
        if add < 0:
            info['stage'][i] = 2  # No goal, go to border
        elif add == carry:
            info['stage'][i] = 1  # Can directly perform goal action
        else:
            info['stage'][i] = 0  # Need to go to border first
    # Group goals by x-y location
    info['loc2goal'] = dict()
    for g in goals:
        if g[0] == -1:
            continue
        loc = (g[1], g[2])
        if loc in info['loc2goal']:
            info['loc2goal'][loc].append(g)
        else:
            info['loc2goal'][loc] = [g]
    # Sort goals into the unique ordering: add low lv -> add high lv -> remove high lv -> remove low lv
    for loc in info['loc2goal']:
        info['loc2goal'][loc].sort(key=lambda g: (-g[0], g[0] * g[3], (g[0] - 1) * g[3]))
    # Mapping from goal id to its order in the unique ordering
    info['id2order'] = dict()
    for g in goals:
        if g[0] != -1:
            loc = (g[1], g[2])
            info['id2order'][g[4]] = info['loc2goal'][loc].index(g)
    # Mark possible heights for each x-y location
    info['loc2height'] = dict()

    for loc in info['loc2goal']:
        info['loc2height'][loc] = set()
        for g in info['loc2goal'][loc]:
            if g[0]:
                info['loc2height'][loc].add(g[3] + 1)
            else:
                info['loc2height'][loc].add(g[3])
            info['loc2height'][loc].add(height[loc[0], loc[1]])
    return info


'''Action validation'''
def movable(x1, y1, x2, y2, height):
    """Valid to move from (x1, y1) to (x2, y2) iff height difference <= 1"""
    # Should use next_height[x2, y2], but if there's a height change at (x2, y2), it'll cause a conflict, thus simplify
    return abs(height[x1, y1] - height[x2, y2]) <= 1

def init_movable(x1, y1, x2, y2, height, goal_info):
    """At initial run, valid to move from (x1, y1) to (x2, y2) iff height difference <= 1 for any possible height"""
    heights1 = goal_info['loc2height'][(x1, y1)] if (x1, y1) in goal_info['loc2height'] else [height[x1, y1]]
    heights2 = goal_info['loc2height'][(x2, y2)] if (x2, y2) in goal_info['loc2height'] else [height[x2, y2]]
    for h1 in heights1:
        for h2 in heights2:
            if abs(h1 - h2) <= 1:
                return True
    return False

def goal_ready(height, goal, x, y, goal_info, init=False):
    """Check if the goal (block) action is ready to be performed"""
    add, gx, gy, lv, _ = goal
    if not next_to_goal(x, y, gx, gy):  # Agent should be next to goal in x-y plane
        return False
    if init:  # Initial run: only need to check if there exists a correct neighbor height
        hs = goal_info['loc2height'][(x, y)] if (x, y) in goal_info['loc2height'] else [height[x, y]]
        for h in hs:
            if h == lv:
                return True
        return False
    else:
        h = height[x, y]
        if h != lv:  # Height of agent location (neighbor to goal) should match goal level
            return False
        gh = height[gx, gy]
        if (add and gh != lv) or (not add and gh != lv + 1):  # Height of goal location should match goal level
            return False
        return True

def next_to_goal(x, y, gx, gy):
    return abs(x - gx) + abs(y - gy) == 1

def get_curr_height(t, t2hid, heights, heights_done, stage):
    """Get height map at current time step"""
    if t in t2hid:
        hid = t2hid[t]
    else:
        hid = -1
    if stage >= 2:
        return heights_done[hid]
    else:
        return heights[hid]

def path2border(env, height, x, y):
    """Check if there is a valid path from (x, y) to border, with DFS"""
    queue = deque()
    visited = set()
    queue.append((x, y))
    while len(queue) > 0:
        x, y = queue.pop()
        if env.border[x, y] == 1:
            return True
        if (x, y) not in visited:
            visited.add((x, y))
            h = height[x, y]
            for (x2, y2) in env.valid_neighbor[(x, y)]:
                if (x2, y2) not in queue and abs(h - height[x2, y2]) <= 1:
                    queue.append((x2, y2))
    # If no path found, return all visited locations
    return False


'''Heuristic'''
def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic(env, stage, x, y, gx, gy):
    """
    Heuristic function to estimate cost to goal
    A* mode
        0: Multi-step A*, use distance to current stage goal
        1: Multi-label A*, use distance to end goal
    """
    w1 = w2 = env.w
    if a_star_mode == 0:
        if stage == 0:  # Distance to depo
            return min(x, w1 - x - 1, y, w2 - y - 1) + 1
        elif stage == 1:  # Distance to goal location
            if x != gx or y != gy:
                return manhattan(x, y, gx, gy)
            else:
                return 2
        else:  # Distance to border
            return min(x, w1 - x - 1, y, w2 - y - 1)
    else:
        if stage == 0:  # Distance to depo to goal to border
            d = 999
            for (nx, ny) in env.valid_neighbor[gx, gy]:  # Will move to a neighbor of goal
                dt = x + 1 + abs(y - ny) + nx  # Via top depo
                dl = y + 1 + abs(x - nx) + ny  # Via left depo
                db = w1 - x + abs(y - ny) + w1 - nx - 1
                dr = w2 - y + abs(x - nx) + w2 - ny - 1
                d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)  # Distance from this neighbor to border
                d = min(d, min(dt, dl, db, dr) + d2)
            return d
        elif stage == 1:  # Distance to goal to depo
            d = 999
            for (nx, ny) in env.valid_neighbor[gx, gy]:
                d1 = manhattan(x, y, nx, ny) + 1
                d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                d = min(d, d1 + d2)
            return d
        else:  # Distance to depo
            return min(x, w1 - x - 1, y, w2 - y - 1)


'''Single agent path finding'''
def filter_constraints(constraints, aid):
    """Filter constraints for a single agent by time"""
    cons = dict()
    cons['range'] = []
    max_step = 0
    for c in constraints:
        if c['agent'] == aid:
            if c['type'] == 'edge2':  # Special edge constraint: all time steps starting from t
                cons['range'].append(c)
            else:
                time = c['time']
                if time not in cons:
                    cons[time] = [c]
                elif c not in cons[time]:
                    cons[time].append(c)
                max_step = max(max_step, time)
    return cons, max_step

def constrained(x1, y1, x2, y2, t, action, constraints):
    """Check if the action is constrained"""
    if t in constraints:
        for c in constraints[t]:
            # Vertex constraint
            if c['type'] == 'vertex' and (x2, y2) == c['loc']:
                return True
            # Edge constraint
            if action == 'move' and c['type'] == 'edge' and ((x1, y1), (x2, y2)) == c['loc']:
                return True
            # Block constraint
            if action == 'goal' and c['type'] == 'block':
                return True
            # Block-edge constraint
            if action == 'goal' and c['type'] == 'block-edge' and (x1, y1) == c['loc']:
                return True
    if action == 'move':  # Edge constraint 2
        for c in constraints['range']:
            if t >= c['time'] and ((x1, y1), (x2, y2)) == c['loc']:
                return True
    return False

def push_node(open_list, node):
    """
    Push a node to the open list
    A* mode
        0: Multi-step A*, order = - stage, g + h, h, x, y
        1: Multi-label A*, order = g + h, h, x, y, - stage
     """
    if a_star_mode == 0:
        heapq.heappush(open_list, (- node.stage, node.g + node.h, node.h, node.x, node.y, node))
    else:
        heapq.heappush(open_list, (node.g + node.h, node.h, node.x, node.y, - node.stage, node))

def a_star(env, goal_info, locations, constraints, heights, t2hid, aid, limit=float('inf')):
    """
    A* search for a single-agent path: finish a goal + return to border
    Node stages:
        0: move to border and interact with depo
        1: finish goal
        2: move to border
    """
    goal = goal_info['goals'][aid]
    add, gx, gy, lv, _ = goal
    ax, ay = locations[aid]
    stage = goal_info['stage'][aid]
    cons, max_con_step = filter_constraints(constraints, aid)

    # If the goal cannot be performed after all height changes, update time limit based on the last change
    # if not path2border(env, heights[-1], gx, gy):
    #     limit = min(limit, max(t2hid.keys()))

    # TODO: if this a problem?
    # if limit < max_con_step:
    #     print("warning!!!")

    # Get the height maps after the goal is finished
    delta = 1 if add else -1
    heights_done = deepcopy(heights)
    for i in range(len(heights)):
        heights_done[i][gx, gy] += delta

    # init_run = len(constraints) == 0
    init_run = True
    open_list = []
    closed_list = set()
    root = Node(None, 0, heuristic(env, stage, ax, ay, gx, gy), ax, ay, stage)
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        x, y, stage, g = node.x, node.y, node.stage, node.g

        '''Completion check: at stage 2, at border, no further constraints'''
        if stage == 2 and (x, y) in env.border_loc and g >= max_con_step:
            return get_path(node)

        '''Expand node'''
        height = get_curr_height(g, t2hid, heights, heights_done, stage)  # Height map at current time
        children = []

        '''Stage 0: try to interact with depo'''
        if stage == 0 and (x, y) in env.border_loc:
            h_val = heuristic(env, 1, x, y, gx, gy)
            child = Node(node, g + 1, h_val, x, y, 1, action='depo')
            children.append(child)

        '''Stage 1: try to finish goal'''
        if stage == 1 and goal_ready(height, goal, x, y, goal_info, init_run) \
                and not constrained(x, y, x, y, g + 1, 'goal', cons):
            h_val = heuristic(env, 2, x, y, gx, gy)
            child = Node(node, g + 1, h_val, x, y, 2, action='goal')
            children.append(child)

        '''Move & stay actions'''
        for (x2, y2) in env.valid_next_loc[(x, y)]:
            if init_run and not init_movable(x, y, x2, y2, height, goal_info):
                continue
            if not init_run and not movable(x, y, x2, y2, height):
                continue
            if constrained(x, y, x2, y2, g + 1, 'move', cons):
                continue
            h_val = heuristic(env, stage, x2, y2, gx, gy)
            child = Node(node, g + 1, h_val, x2, y2, stage)
            children.append(child)

        '''Push children'''
        for child in children:
            key = (child.stage, child.x, child.y, child.g)
            if key not in closed_list and child.g <= limit:
                push_node(open_list, child)
                closed_list.add(key)
    return None, None

def get_path(node):
    """Get a successful single-agent path to the goal, plus the time that goal is finished"""
    path = []
    t, seen = 0, False
    while node:
        path.append((node.x, node.y, node.action))
        if node.action == 'goal':
            seen = True
        if not seen:
            t += 1
        node = node.parent
    path.reverse()
    t = len(path) - t - 1
    return path, t


class Node:
    def __init__(self, parent, g, h, x, y, stage, action='move'):
        self.parent = parent
        self.g = g
        self.h = h
        self.x = x
        self.y = y
        self.stage = stage
        self.action = action
