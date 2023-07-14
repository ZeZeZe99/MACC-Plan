import heapq

"""
Low level A* search for single agent path, in 3D space

A* mode
    0: Multi-step A* (greedily optimize each sub-path in order)
    1: Multi-label A* (optimize the whole path)
    2: Multi-label A* with fuel consumption (favor paths with less fuel consumption)
"""

a_star_mode = 2


'''Goal processing'''
def process_goal(env, goals, carry_stats):
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
    # Mark possible heights for each x-y location
    info['loc2height'] = dict()
    for loc in env.valid_loc:
        loc2height = set()
        loc2height.add(env.height[loc[0], loc[1]])
        if loc in info['loc2goal']:
            for g in info['loc2goal'][loc]:
                if g[0]:
                    loc2height.add(g[3] + 1)
                else:
                    loc2height.add(g[3])
        info['loc2height'][loc] = loc2height
    return info


'''Action validation'''
def goal_ready(goal, x, y, z):
    """Check if the goal (block) action is ready to be performed"""
    add, gx, gy, lv, _ = goal
    if manhattan(x, y, gx, gy) != 1:  # Agent should be next to goal in x-y plane
        return False
    return z == lv


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
            if c['type'] == 'lv-edge' and c['range']:
                cons['range'].append(c)
            elif c['type'] == 'lv-vertex' and c['range']:
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
    """Check if the action is constrained in 2D space"""
    if t in constraints:
        for c in constraints[t]:
            # Vertex constraint
            if c['type'] == 'vertex' and (x2, y2) == c['loc']:
                return True
            # Edge constraint
            if action == 'move':
                if c['type'] == 'edge' and ((x1, y1), (x2, y2)) == c['loc']:
                    return True
            # Block constraint
            if action == 'goal' and c['type'] == 'block':
                return True
    return False

def constrained_3d(x2, y2, z2, t, constraints):
    """Check if the action is constrained in 3D space"""
    if t in constraints:
        for c in constraints[t]:
            if c['type'] == 'lv-vertex' and ((x2, y2), z2) == c['loc']:
                return True
    for c in constraints['range']:
        if t >= c['time'] and ((x2, y2), z2) == c['loc']:
            return True
    return False

def push_node(open_list, node):
    """
    Push a node to the open list
    A* mode
        0: Multi-step A*, order = - stage, g + h, h, x, y, z
        1: Multi-label A*, order = g + h, h, x, y, z, - stage
        2: Multi-label A* + fuel consumption, order = g + h, h, fuel, x, y, z, - stage
     """
    if a_star_mode == 0:
        heapq.heappush(open_list, (- node.stage, node.g + node.h, node.h, node.x, node.y, node.z, node))
    elif a_star_mode == 1:
        heapq.heappush(open_list, (node.g + node.h, node.h, node.x, node.y, node.z, - node.stage, node))
    else:
        heapq.heappush(open_list, (node.g + node.h, node.h, node.fuel, node.x, node.y, node.z, - node.stage, node))

def a_star(env, goal_info, locations, constraints, aid, limit=float('inf')):
    """
    A* search for a single-agent path in 3D space: finish a goal + return to border
    Node stages:
        0: move to border and interact with depo (if needed)
        1: finish goal
        2: move to border
    """
    goal = goal_info['goals'][aid]
    add, gx, gy, lv, _ = goal
    ax, ay, az = locations[aid]
    stage = goal_info['stage'][aid]
    cons, max_con_step = filter_constraints(constraints, aid)

    open_list = []
    closed_list = set()
    root = Node(None, 0, heuristic(env, stage, ax, ay, gx, gy), ax, ay, az, stage)
    push_node(open_list, root)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        x, y, z, stage, g = node.x, node.y, node.z, node.stage, node.g

        '''Completion check: at stage 2, at border, no further constraints'''
        if stage == 2 and (x, y) in env.border_loc and g >= max_con_step:
            return get_path(node)

        '''Expand node'''
        children = []

        '''Stage 0: try to interact with depo'''
        if stage == 0 and (x, y) in env.border_loc and not constrained(x, y, x, y, g + 1, 'depo', cons):
            h_val = heuristic(env, 1, x, y, gx, gy)
            child = Node(node, g + 1, h_val, x, y, z, 1, action='depo')
            children.append(child)

        '''Stage 1: try to finish goal'''
        if stage == 1 and goal_ready(goal, x, y, z) and not constrained(x, y, x, y, g + 1, 'goal', cons) \
                and not constrained_3d(x, y, z, g + 1, cons):
            h_val = heuristic(env, 2, x, y, gx, gy)
            child = Node(node, g + 1, h_val, x, y, z, 2, action='goal')
            children.append(child)

        '''Move & stay actions'''
        for (x2, y2) in env.valid_next_loc[(x, y)]:
            if constrained(x, y, x2, y2, g + 1, 'move', cons):
                continue
            zs = goal_info['loc2height'][(x2, y2)]
            for z2 in zs:
                if x == x2 and y == y2 and z != z2:  # Skip vertical move
                    continue
                if abs(z - z2) > 1 or constrained_3d(x2, y2, z2, g + 1, cons):  # Skip move with height difference > 1
                    continue
                h_val = heuristic(env, stage, x2, y2, gx, gy)
                child = Node(node, g + 1, h_val, x2, y2, z2, stage)
                children.append(child)

        '''Push children & duplicate detection'''
        for child in children:
            key = (child.stage, child.x, child.y, child.z, child.g)
            if key not in closed_list and child.g <= limit:  # Trim nodes with g > limit
                push_node(open_list, child)
                closed_list.add(key)
    return None, None

def get_path(node):
    """
    Get a successful single-agent path to the goal, plus the time that goal is finished
    Result node in path: ((x, y, z), action)
    """
    path = []
    t, seen = 0, False
    while node:
        path.append(((node.x, node.y, node.z), node.action))
        if node.action == 'goal':
            seen = True
        if not seen:
            t += 1
        node = node.parent
    path.reverse()
    t = len(path) - t - 1
    return path, t


class Node:
    def __init__(self, parent, g, h, x, y, z, stage, action='move'):
        self.parent = parent
        self.g = g
        self.h = h
        self.x = x
        self.y = y
        self.z = z
        self.stage = stage
        self.action = action

        # Fuel consumption = 1 for each non-stay action
        if parent:
            self.fuel = parent.fuel
            if x != parent.x or y != parent.y or action != 'move':
                self.fuel += 1
        else:
            self.fuel = 0
