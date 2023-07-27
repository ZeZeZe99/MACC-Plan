import heapq
from collections import deque
import numpy as np

"""
Low level A* search to plan a single agent path in 3D space to
    1) finish goal
    2) return to border (parking location)

A* mode
    0: order = -stage, f, h 
        Favors reaching later stages earlier, similar to multi-step A* with back-track
        h = cost to goal of current stage
    1: order = f, h
        Favors finishing goal and returning to border earlier, similar to multi-label A*
        h = cost to goal then back to border
    2: order = f, h, h2g, (fuel)
        Similar to 1, favors finishing goal earlier (2nd)
        h = cost to goal then back to border, h2g = cost to goal
    3: order = f, h2g, h, (fuel)
        Favors finishing goal earlier, then favors returning to border earlier
    4: order = f, collision, h, h2g, (fuel)
        Favors fewer collisions (1st), returning to border earlier (2nd), finishing goal earlier (3rd)
        collision = # collisions with existing paths
    Extra: favors less fuel consumption (non-stay actions)

Heuristic mode
    0: Manhattan distance to goal of current stage
    1: Distance to the end goal, under-estimate
    2: Distance to the end goal, pre-computed
    3: Teleport, have to exit world
"""

a_star_mode = 4
heuristic_mode = 3
teleport = heuristic_mode == 3

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
    # Assumption: only has (add low lv -> add high lv), or (remove high lv -> remove low lv)
    for loc in info['loc2goal']:
        info['loc2goal'][loc].sort(key=lambda g: (-g[0], g[0] * g[3], (g[0] - 1) * g[3]))
    # Find the order of the goal in its location
    info['goal2order'] = dict()
    for g in goals:
        if g[0] != -1:
            info['goal2order'][g] = info['loc2goal'][(g[1], g[2])].index(g)
    # Mark possible heights for each x-y location
    info['loc2height'] = dict()
    for loc in env.valid_loc:
        loc2height = []
        loc2height.append(env.height[loc[0], loc[1]])
        if loc in info['loc2goal']:
            for g in info['loc2goal'][loc]:
                if g[0]:
                    loc2height.append(g[3] + 1)
                else:
                    loc2height.append(g[3])
        info['loc2height'][loc] = loc2height
    # Mark possible neighbors for each goal (where the agent can stand at while performing the goal action)
    info['goal2neighbor'] = dict()
    for _, gx, gy, lv, _ in goals:
        if gx != -1:
            info['goal2neighbor'][(gx, gy, lv)] = set()
            for (nx, ny) in env.valid_neighbor[gx, gy]:
                nzs = info['loc2height'][(nx, ny)]
                if lv in nzs:
                    info['goal2neighbor'][(gx, gy, lv)].add((nx, ny, lv))
    return info


'''Heuristic'''
def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def distance2border(env, loc2height):
    """Pre-compute the shortest distance to border for each 3D location"""
    distance = dict()
    open_list = deque()
    for (x, y) in env.border_loc:
        distance[(x, y, 0)] = 0
        open_list.append((x, y, 0))
    while open_list:
        x, y, z = open_list.popleft()
        d = distance[(x, y, z)]
        for (nx, ny) in env.valid_neighbor[x, y]:
            nzs = loc2height[nx, ny]
            for nz in nzs:
                if abs(nz - z) > 1:
                    continue
                if (nx, ny, nz) not in distance:
                    distance[(nx, ny, nz)] = d + 1
                    open_list.append((nx, ny, nz))
    return distance

def distance2neighbor(env, loc2height, goal2neighbor):
    """
    Pre-compute the shortest distance to neighbor of each goal via depo
    Mirror map is a "+" shape map, with the original map at the center, and its flip versions along 4 edges
    Distance from a location to a neighbor via left depo = distance from its left mirror to the neighbor
    """
    w = env.w
    d2neighbor = dict()
    for g, nbs in goal2neighbor.items():
        gx, gy, lv = g
        for (nx, ny, nz) in nbs:
            if (nx, ny, lv) in d2neighbor:
                continue
            distance = dict()
            distance[(nx + w, ny + w, lv)] = 0
            open_list = deque()
            open_list.append((nx + w, ny + w, lv))
            while open_list:
                x, y, z = open_list.popleft()
                d = distance[(x, y, z)]
                for (x2, y2) in env.mirror_neighbor[x, y]:
                    zs = loc2height[env.mirror2origin[(x2, y2)]]
                    for z2 in zs:
                        if abs(z2 - z) > 1:
                            continue
                        if (x2, y2, z2) not in distance:
                            distance[(x2, y2, z2)] = d + 1
                            open_list.append((x2, y2, z2))
            d2neighbor[(nx, ny, lv)] = distance
    return d2neighbor

def heuristic(env, goal_info, stage, x, y, z, gx, gy, lv):
    """
    Heuristic function to estimate cost to goal
    Heuristic mode
        0: distance to current stage goal
        1: distance to end goal (border)
        2: use distance to end goal (border), pre-compute
    Return (h_val, h_val to goal action)
    """
    w1 = w2 = env.w
    if heuristic_mode == 0:
        if stage == 0:  # Distance to depo
            return min(x, w1 - x - 1, y, w2 - y - 1) + 1, 0
        elif stage == 1:  # Distance to goal location
            if x != gx or y != gy:
                return manhattan(x, y, gx, gy), 0
            else:
                return 2, 0
        else:  # Distance to border
            return min(x, w1 - x - 1, y, w2 - y - 1), 0
    elif heuristic_mode == 1:
        if stage == 0:  # Distance to depo to goal to border
            h = 999
            hg = 0
            for (nx, ny) in env.valid_neighbor[gx, gy]:  # Will move to a neighbor of goal
                d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)  # Distance from this neighbor to border
                # Distance to a depo + distance to neighbor + 'depo' and 'goal' actions
                d1 = manhattan(x, y, nx, ny) + 2 * min(x, nx, y, ny, w1 - x - 1, d2) + 2
                d = d1 + d2
                if d < h:
                    h = d
                    hg = d1
        elif stage == 1:  # Distance to goal to depo
            h = 999
            hg = 0
            for (nx, ny) in env.valid_neighbor[gx, gy]:
                d1 = manhattan(x, y, nx, ny) + 1
                d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                d = d1 + d2
                if d < h:
                    h = d
                    hg = d1
        else:  # Distance to border
            h = min(x, w1 - x - 1, y, w2 - y - 1)
            hg = 0
        return h, hg
    elif heuristic_mode == 2:
        if stage == 0:  # Distance to depo to goal to border
            h = 1000
            hg = 0
            for (nx, ny, nz) in goal_info['goal2neighbor'][(gx, gy, lv)]:
                d2nbr = goal_info['d2neighbor'].get((nx, ny, nz))
                if d2nbr is None:
                    continue
                d1 = 1000
                for (mx, my) in env.origin2mirror[(x, y)]:  # Distance to neighbor via a depo
                    d1 = min(d1, d2nbr.get((mx, my, z), 1000))
                d2 = goal_info['d2border'].get((nx, ny, nz), 1000)
                d = d1 + 1 + d2
                if d < h:
                    h = d
                    hg = d1 + 1
        elif stage == 1:  # Distance to goal to depo
            h = 1000
            hg = 0
            for (nx, ny, nz) in goal_info['goal2neighbor'][(gx, gy, lv)]:
                d2nbr = goal_info['d2neighbor'].get((nx, ny, nz))
                if d2nbr is None:
                    continue
                d1 = d2nbr.get((x + w1, y + w2, z), 1000)  # d to neighbor directly
                d2 = goal_info['d2border'].get((nx, ny, nz), 1000)
                d = d1 + 1 + d2
                if d < h:
                    h = d
                    hg = d1 + 1
        else:  # Distance to border
            h = goal_info['d2border'].get((x, y, z), 1000)
            hg = 0
        return h, hg
    elif heuristic_mode == 3:
        if stage == 0:
            d1 = 2  # 'depo' and move into world
            if x != -1:
                d1 += min(x, w1 - x - 1, y, w2 - y - 1) + 1  # Move to border + leave world
            d2 = 999
            for (nx, ny) in env.valid_neighbor[gx, gy]:
                d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1, d2)
            d = d1 + d2 * 2 + 1
        elif stage == 1:
            if x == -1:
                d1 = 999
                for (nx, ny) in env.valid_neighbor[gx, gy]:
                    d1 = min(d1, nx, w1 - nx - 1, ny, w2 - ny - 1)
                d = 2 * d1 + 2
            else:
                d = 999
                for (nx, ny) in env.valid_neighbor[gx, gy]:
                    d1 = min(d, manhattan(x, y, nx, ny))
                    d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                    d = min(d, d1 + d2 + 1)
        else:
            d = min(x, w1 - x - 1, y, w2 - y - 1)
        return d, 0


    else:
        raise NotImplementedError


'''Action validation'''
def goal_ready(goal, x, y, z):
    """Check if the goal (block) action is ready to be performed"""
    add, gx, gy, lv, _ = goal
    if manhattan(x, y, gx, gy) != 1:  # Agent should be next to goal in x-y plane
        return False
    return z == lv


'''Single agent path finding'''
def filter_constraints(constraints, aid):
    """Filter constraints for a single agent by time. Constraint = (type, agent, time, loc, range)"""
    cons = dict()
    cons['range'] = []
    max_step = 0
    for c in constraints:
        if c[1] == aid:
            if c[0] == 'lv-vertex' and c[4]:
                cons['range'].append(c)
            else:
                time = c[2]
                if time not in cons:
                    cons[time] = [c]
                elif c not in cons[time]:
                    cons[time].append(c)
                max_step = max(max_step, time)
    return cons, max_step

def constrained(x1, y1, x2, y2, t, action, constraints):
    """Check if the action is constrained in 2D space. Constraint = (type, agent, time, loc, range)"""
    if t in constraints:
        for c in constraints[t]:
            # Vertex constraint
            if c[0] == 'vertex' and (x2, y2) == c[3]:
                return True
            # Edge constraint
            if action == 'move':
                if c[0] == 'edge' and ((x1, y1), (x2, y2)) == c[3]:
                    return True
            # Block constraint
            if action == 'goal' and c[0] == 'block':
                return True
    return False

def constrained_3d(x2, y2, z2, t, constraints):
    """Check if the action is constrained in 3D space. Constraint = (type, agent, time, loc, range)"""
    if t in constraints:
        for c in constraints[t]:
            if c[0] == 'lv-vertex' and ((x2, y2), z2) == c[3]:
                return True
    for c in constraints['range']:
        if t >= c[2] and ((x2, y2), z2) == c[3]:
            return True
    return False

def construct_heights(height, block_actions, goal_info=None, ignore=None):
    """
    Construct the height map sequence from block actions, plus a mapping from t to height map id, for conflict detection
    Block action: (t, (add, x, y, lv, gid))
    Assume block action at t will change the map at t+1
    Assume goal actions at the same x-y location are performed in order
    When ignore is not None, ignore the specified goal action, but consider its effect when its next goal is performed
    """
    ignore_g = next_g = None
    offset = 0
    if ignore is not None and block_actions[ignore][0] >= 0:
        ignore_g = block_actions[ignore][1]
        gs = goal_info['loc2goal'][ignore_g[1:3]]
        order = goal_info['goal2order'][ignore_g]
        if order != len(gs) - 1:
            offset = 1 if ignore_g[0] == 1 else -1
            next_g = gs[order + 1]

    block_actions = block_actions.copy()
    block_actions.sort()  # Sort by time
    unique = set([b[0] for b in block_actions])  # Unique time steps
    unique.discard(-1)  # Ignore dummy action
    heights = np.tile(height, (len(unique) + 1, 1, 1))  # Height map sequence, include initial one
    t2hid = {0: 0}
    prev_t, hid = 0, 0

    for t, g in block_actions:  # t will always start above 0
        add, x, y, lv, _ = g
        if ignore_g == g:
            continue
        else:
            if t < 0:  # Skip dummy actions
                continue
            if t > prev_t:
                hid += 1
            delta = 1 if add else -1
            if next_g == g:
                delta += offset
            heights[hid:, x, y] += delta
        # Update mapping from t to height map id
        if t != prev_t:
            for i in range(prev_t, t):
                t2hid[i + 1] = hid - 1
            t2hid[t + 1] = hid
        prev_t = t
    return heights, t2hid

def count_collisions(node, aid, paths, heights, t2hid, block_actions):
    """Count the number of collisions with other paths"""
    collision = 0
    time = node.g
    if time >= len(paths[0]):
        time = len(paths[0]) - 1
    height = heights[t2hid[time]] if time in t2hid else heights[-1]
    x, y, z = node.x, node.y, node.z
    px, py = node.parent.x, node.parent.y
    if x == -1:
        return collision
    for i in range(len(paths)):
        if i == aid:
            continue
        x2, y2, z2 = paths[i][time][0]
        px2, py2, _ = paths[i][time - 1][0]
        if x2 == -1:
            continue
        if x == x2 and y == y2:
            collision += 1
        if x == px2 and y == py2 and px == x2 and py == y2:
            collision += 1
        t, (_, gx, gy, _, _) = block_actions[i]
        if t == time and ((x, y) == (gx, gy) or (px, py) == (gx, gy)):
            collision += 1
        if z != height[x, y]:
            collision += 1
    return collision

def push_node(open_list, node, gen):
    """
    Push a node to the open list
    A* mode
        0: order = - stage, g + h, h
        1: order = g + h, h
        2: order = g + h, h, h2g, fuel
        3: order = g + h, h2g, h, fuel
        4: order = g + h, collision, h, h2g, fuel
     """
    g, h, x, y, z, stage = node.g, node.h, node.x, node.y, node.z, node.stage
    h2g, collision, fuel = node.h2g, node.collision, node.fuel
    if a_star_mode == 0:
        heapq.heappush(open_list, (- stage, g + h, h, x, y, z, node))
    elif a_star_mode == 1:
        heapq.heappush(open_list, (g + h, h, x, y, z, node))
    elif a_star_mode == 2:
        heapq.heappush(open_list, (g + h, h, h2g, fuel, x, y, z, node))
    elif a_star_mode == 3:
        heapq.heappush(open_list, (g + h, h2g, h, fuel, x, y, z, node))
    elif a_star_mode == 4:
        heapq.heappush(open_list, (g + h, collision, h, h2g, fuel, x, y, z, gen, node))
    else:
        raise NotImplementedError

def a_star(env, goal_info, locations, constraints, aid, limit=float('inf'), paths=None, block_actions=None):
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

    if paths is not None and a_star_mode == 4:
        heights, t2hid = construct_heights(env.height, block_actions, goal_info, ignore=aid)
    else:
        heights, t2hid = None, None

    open_list = []
    closed_list = dict()
    gen = 0
    h_val, h2g = heuristic(env, goal_info, stage, ax, ay, az, gx, gy, lv)
    root = Node(None, 0, h_val, h2g, ax, ay, az, stage)
    push_node(open_list, root, gen)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        x, y, z, stage, g = node.x, node.y, node.z, node.stage, node.g

        '''Completion check: at stage 2, at border, no further constraints'''
        if stage == 2 and (x, y) in env.border_loc and g >= max_con_step:
            return get_path(node)

        '''Expand node'''
        children = []

        '''Stage 0: try to interact with depo'''
        if stage == 0:
            if teleport and x == -1:
                h_val, h2g = heuristic(env, goal_info, 1, x, y, z, gx, gy, lv)
                child = Node(node, g + 1, h_val, h2g, x, y, z, 1, action='depo')
                children.append(child)
            elif not teleport and (x, y) in env.border_loc and not constrained(x, y, x, y, g + 1, 'depo', cons):
                h_val, h2g = heuristic(env, goal_info, 1, x, y, z, gx, gy, lv)
                child = Node(node, g + 1, h_val, h2g, x, y, z, 1, action='depo')
                children.append(child)

        '''Stage 1: try to finish goal'''
        if stage == 1 and goal_ready(goal, x, y, z) and not constrained(x, y, x, y, g + 1, 'goal', cons) \
                and not constrained_3d(x, y, z, g + 1, cons):
            h_val, h2g = heuristic(env, goal_info, 2, x, y, z, gx, gy, lv)
            child = Node(node, g + 1, h_val, h2g, x, y, z, 2, action='goal')
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
                # Skip level inaccessible based on own goal action, assume no 2 goals for the same block
                if x2 == gx and y2 == gy:
                    if stage == 2 and ((add and z2 <= lv) or (not add and z2 >= lv + 1)):
                        continue
                    if stage != 2 and ((add and z2 >= lv + 1) or (not add and z2 <= lv)):
                        continue
                h_val, h2g = heuristic(env, goal_info, stage, x2, y2, z2, gx, gy, lv)
                child = Node(node, g + 1, h_val, h2g, x2, y2, z2, stage)
                children.append(child)

        '''Teleport: leave world & enter world'''
        if teleport:
            if (x, y) in env.border_loc:
                h_val, h2g = heuristic(env, goal_info, stage, -1, -1, -1, gx, gy, lv)
                child = Node(node, g + 1, h_val, h2g, -1, -1, -1, stage)
                children.append(child)
            elif x == -1:
                for (x2, y2) in env.border_loc:
                    if constrained(x, y, x2, y2, g + 1, 'move', cons):
                        continue
                    h_val, h2g = heuristic(env, goal_info, stage, x2, y2, 0, gx, gy, lv)
                    child = Node(node, g + 1, h_val, h2g, x2, y2, 0, stage)
                    children.append(child)

        '''Push children & duplicate detection'''
        for child in children:
            if child.g <= limit:  # Trim nodes with g > limit
                if paths is not None and a_star_mode == 4:
                    child.collision += count_collisions(child, aid, paths, heights, t2hid, block_actions)
                key = (child.stage, child.x, child.y, child.z, child.g)
                val = (child.collision, child.fuel)
                # New node, fewer collisions, or less fuel
                if key not in closed_list or closed_list[key] > val:
                    gen += 1
                    push_node(open_list, child, gen)
                    closed_list[key] = val

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
    def __init__(self, parent, g, h, h2g, x, y, z, stage, action='move'):
        self.parent = parent
        self.g = g
        self.h = h
        self.h2g = h2g
        self.x = x
        self.y = y
        self.z = z
        self.stage = stage
        self.action = action

        if parent:
            self.collision = parent.collision
            # Fuel consumption = 1 for each non-stay action
            self.fuel = parent.fuel
            if x != parent.x or y != parent.y or action != 'move':
                self.fuel += 1
        else:
            self.collision = 0
            self.fuel = 0
