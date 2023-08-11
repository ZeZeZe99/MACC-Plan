import heapq
import numpy as np
from collections import deque

"""
Low level A* search to plan a single agent path in 3D space to
    1) finish goal
    2) return to parking location

A* ordering mode
    0: order = -stage, f, h 
        Favors reaching later stages earlier, similar to multi-step A* with back-track
        h = cost to goal of current stage
    1: order = f, h
        Favors parking earlier, similar to multi-label A*
        h = cost to goal then back to park location
    2: order = f, h, h2g, (fuel)
        Similar to 1, favors finishing goal earlier (2nd)
        h = cost to goal then back to park location, h2g = cost to goal
    3: order = f, h2g, h, (fuel)
        Favors finishing goal earlier (1st), then parking earlier (2nd)
    4: order = f, collision, h, h2g, (fuel)
        Favors fewer collisions (1st), parking earlier (2nd), finishing goal earlier (3rd)
        collision = # collisions with existing paths
    5: order = f, collision, h, h2g, in_world, (fuel)
        Similar to 4, favors less in-world actions (4th)
    Extra: favors less fuel consumption (non-stay actions)

Heuristic mode
    0: Manhattan distance to goal of current stage
    1: Manhattan distance to the end goal
    2: Precomputed distance to the end goal
"""

'''Heuristic'''
def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def distance2border(env, loc2height):
    """Pre-compute the shortest distance to border, considering all possible heights"""
    distance = dict()
    open_list = deque()
    for (x, y) in env.border_loc:
        distance[(x, y)] = 0
        open_list.append((x, y))
    while open_list:
        x, y = open_list.popleft()
        d = distance[(x, y)]
        zs = loc2height[(x, y)]
        for (nx, ny) in env.valid_neighbor[(x, y)]:
            if (nx, ny) in distance:
                continue
            nzs = loc2height[(nx, ny)]
            skip = False
            for z1 in zs:
                for z2 in nzs:
                    if abs(z1 - z2) <= 1:
                        distance[(nx, ny)] = d + 1
                        open_list.append((nx, ny))
                        skip = True
                        break
                if skip:
                    break
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
        for nx, ny in nbs:
            if (nx, ny) in d2neighbor:
                continue
            distance = dict()
            distance[(nx + w, ny + w)] = 0
            open_list = deque()
            open_list.append((nx + w, ny + w))
            while open_list:
                x, y = open_list.popleft()
                d = distance[(x, y)]
                zs = loc2height[env.mirror2origin[(x, y)]]
                for (x2, y2) in env.mirror_neighbor[x, y]:
                    if (x2, y2) in distance:
                        continue
                    nzs = loc2height[env.mirror2origin[(x2, y2)]]
                    skip = False
                    for z1 in zs:
                        for z2 in nzs:
                            if abs(z1 - z2) <= 1:
                                distance[(x2, y2)] = d + 1
                                open_list.append((x2, y2))
                                skip = True
                                break
                        if skip:
                            break
            d2neighbor[(nx, ny)] = distance
    return d2neighbor

def heuristic(env, goal_info, stage, x, y, gx, gy, lv, mode, teleport):
    """
    Heuristic function to estimate cost to goal
    Heuristic mode
        0: manhattan distance to current stage goal
        1: manhattan distance to end goal (border)
        2: precomputed distance to end goal (border)
    Return (h_val, h_val to goal action)
    """
    w1 = w2 = env.w
    if mode == 0:
        if stage == 0:  # Distance to depo
            return min(x, w1 - x - 1, y, w2 - y - 1) + 1, 0
        elif stage == 1:  # Distance to goal location
            if x != gx or y != gy:
                return manhattan(x, y, gx, gy), 0
            else:
                return 2, 0
        else:  # Distance to border
            return min(x, w1 - x - 1, y, w2 - y - 1), 0
    elif mode == 1:
        h = 999
        hg = 0
        if stage == 0:  # Distance to depo to goal to border
            if teleport:
                d1 = 2  # 'depo' and move into world
                if x != -1:
                    d1 += min(x, w1 - x - 1, y, w2 - y - 1) + 1  # Move to border + leave world
                d2 = 999
                for (nx, ny) in env.valid_neighbor[gx, gy]:
                    d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1, d2)
                h = d1 + d2 * 2 + 2
                hg = d1 + d2 + 1
            else:
                for (nx, ny) in env.valid_neighbor[gx, gy]:  # Will move to a neighbor of goal
                    # Distance from this neighbor to border
                    d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                    # Distance to a depo + distance to neighbor + 'depo' and 'goal' actions
                    d1 = manhattan(x, y, nx, ny) + 2 * min(x, y, w1 - x - 1, w2 - y - 1, d2) + 2
                    d = d1 + d2
                    if d < h:
                        h = d
                        hg = d1
        elif stage == 1:  # Distance to goal to border
            if teleport:
                if x == -1:
                    d1 = 999
                    for (nx, ny) in env.valid_neighbor[gx, gy]:
                        d1 = min(d1, nx, w1 - nx - 1, ny, w2 - ny - 1)
                    h = 2 * d1 + 3
                    hg = d1 + 2
                else:
                    for (nx, ny) in env.valid_neighbor[gx, gy]:
                        d1 = manhattan(x, y, nx, ny)
                        d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                        if d1 + d2 + 2 < h:
                            h = d1 + d2 + 2
                            hg = d1 + 1
            else:
                for (nx, ny) in env.valid_neighbor[gx, gy]:
                    d1 = manhattan(x, y, nx, ny) + 1
                    d2 = min(nx, w1 - nx - 1, ny, w2 - ny - 1)
                    d = d1 + d2
                    if d < h:
                        h = d
                        hg = d1
        else:  # Distance to parking location
            h = min(x, w1 - x - 1, y, w2 - y - 1) + teleport
            hg = 0
        return h, hg
    elif mode == 2:
        h = 1000
        hg = 0
        if stage == 0:  # Distance to depo to goal to parking location
            if teleport:
                d1 = 2  # 'depo' and move into world
                if x != -1:
                    d1 += goal_info['d2border'].get((x, y), 1000) + 1
                d2 = 1000
                for (nx, ny) in goal_info['g2neighbor'][(gx, gy, lv)]:
                    d2 = min(d2, goal_info['d2border'].get((nx, ny), 1000))
                h = d1 + d2 * 2 + 2
                hg = d1 + d2 + 1
            else:
                for (nx, ny) in goal_info['g2neighbor'][(gx, gy, lv)]:
                    d2nbr = goal_info['d2neighbor'].get((nx, ny))
                    if d2nbr is None:
                        continue
                    d1 = 1000
                    for (mx, my) in env.origin2mirror[(x, y)]:  # Distance to neighbor via a depo
                        d1 = min(d1, d2nbr.get((mx, my), 1000))
                    d2 = goal_info['d2border'].get((nx, ny), 1000)
                    d = d1 + 1 + d2
                    if d < h:
                        h = d
                        hg = d1 + 1
        elif stage == 1:  # Distance to goal to depo
            if teleport:
                if x == -1:
                    d1 = 1000
                    for (nx, ny) in goal_info['g2neighbor'][(gx, gy, lv)]:
                        d1 = min(d1, goal_info['d2border'].get((nx, ny), 1000))
                    h = 2 * d1 + 3
                    hg = d1 + 2
                else:
                    for (nx, ny) in goal_info['g2neighbor'][(gx, gy, lv)]:
                        d2nbr = goal_info['d2neighbor'].get((nx, ny))
                        if d2nbr is None:
                            continue
                        d1 = d2nbr.get((x + w1, y + w2), 1000)  # d to neighbor directly
                        d2 = goal_info['d2border'].get((nx, ny), 1000)
                        if d1 + d2 + 2 < h:
                            h = d1 + d2 + 2
                            hg = d1 + 1
            else:
                for (nx, ny) in goal_info['g2neighbor'][(gx, gy, lv)]:
                    d2nbr = goal_info['d2neighbor'].get((nx, ny))
                    if d2nbr is None:
                        continue
                    d1 = d2nbr.get((x + w1, y + w2), 1000)  # d to neighbor directly
                    d2 = goal_info['d2border'].get((nx, ny), 1000)
                    d = d1 + 1 + d2
                    if d < h:
                        h = d
                        hg = d1 + 1
        else:  # Distance to parking location
            h = 0 if x == -1 else goal_info['d2border'].get((x, y), 1000) + teleport
            hg = 0
        return h, hg
    else:
        raise NotImplementedError


'''Action validation'''
def movable_neighbor(env, loc2height):
    movable_nbr = dict()
    for loc in env.valid_loc:
        movable_nbr[loc] = []
        for (nx, ny) in env.valid_neighbor[loc]:
            if movable(loc[0], loc[1], nx, ny, loc2height):
                movable_nbr[loc].append((nx, ny))
    movable_nbr[(-1, -1)] = []
    return movable_nbr

def movable(x1, y1, x2, y2, loc2height):
    """At initial run, valid to move from (x1, y1) to (x2, y2) if height difference <= 1 for any possible height"""
    for h1 in loc2height[(x1, y1)]:
        for h2 in loc2height[(x2, y2)]:
            if abs(h1 - h2) <= 1:
                return True
    return False

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


'''Single agent path finding'''
def filter_constraints(constraints, aid):
    """Filter constraints for a single agent by time. Constraint = (type, agent, time, loc, range)"""
    cons = dict()
    cons['range'] = []
    max_step = 0
    for c in constraints:
        if c[1] == aid:
            if c[0] == 'range-edge' or c[0] == 'range-block-nbr':
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
    """Check if the action is constrained"""
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
            # Block neighbor constraint
            if action == 'goal' and c[0] == 'block-nbr' and (x2, y2) == c[3]:
                return True
        for c in constraints['range']:
            if c[2] <= t:
                if c[0] == 'range-edge' and ((x1, y1), (x2, y2)) == c[3]:
                    return True
                if c[0] == 'range-block-nbr' and (x2, y2) == c[3]:
                    return True
    return False

def construct_heights(height, block_actions, ignore_goals=None):
    """
    Construct the height map sequence from block actions, plus a mapping from t to height map id, for conflict detection
    Block action: (t, (add, x, y, lv, gid))
    Assume block action at t will change the map at t+1
    When goal to ignore is given, ignore it and all its successors
    """
    if ignore_goals is None:
        ignore_goals = set()
    block_actions = block_actions.copy()
    block_actions.sort()  # Sort by time
    heights = [height]  # Height map sequence, include initial one
    t2hid = {0: 0}
    prev_t, hid = 0, 0

    for t, g in block_actions:  # t will always start above 0
        add, x, y, lv, _ = g
        if g in ignore_goals:  # Skip ignored goals
            continue
        if t < 0:  # Skip dummy actions
            continue
        if t > prev_t:
            hid += 1
            heights.append(heights[-1].copy())
        heights[-1][x, y] += 1 if add else -1  # Update height map
        if t > prev_t:  # Update mapping from t to height map id
            for i in range(prev_t, t):
                t2hid[i + 1] = hid - 1
            t2hid[t + 1] = hid
        prev_t = t
    heights = np.stack(heights, axis=0)
    return heights, t2hid

def count_collisions(node, aid, paths, heights, t2hid, block_actions, ignore_goals):
    """Count the number of collisions with other paths"""
    collision = 0
    x, y = node.x, node.y
    px, py = node.parent.x, node.parent.y
    if x == -1:  # No collision when out of map
        return collision
    if ignore_goals is None:
        ignore_goals = set()
    time = node.g
    if time >= len(paths[0]):
        time = len(paths[0]) - 1
    for i in range(len(paths)):
        if i == aid:  # Skip own path
            continue
        x2, y2 = paths[i][time][0]
        px2, py2 = paths[i][time - 1][0]
        if x2 == -1:  # Skip if the other agent is out of map
            continue
        if x == x2 and y == y2:  # Vertex collision
            collision += 1
        if x == px2 and y == py2 and px == x2 and py == y2:  # Edge collision
            collision += 1
        t, g = block_actions[i]
        if g in ignore_goals:  # Skip ignored goals for other collisions
            continue
        gx, gy = g[2:4]
        if t == time and ((x, y) == (gx, gy) or (px, py) == (gx, gy)):  # Agent-block collision
            collision += 1
    i = 1 if node.stage == 2 else 0
    height = heights[i, t2hid[time]] if time in t2hid else heights[i, -1]
    if abs(height[px, py] - height[x, y]) > 1:  # Height collision
        collision += 1
    return collision

def push_node(open_list, node, gen, mode):
    """
    Push a node to the open list
    A* mode
        0: order = - stage, g + h, h
        1: order = g + h, h
        2: order = g + h, h, h2g, fuel
        3: order = g + h, h2g, h, fuel
        4: order = g + h, collision, h, h2g, fuel
        5: order = g + h, collision, h, h2g, in_world
     """
    g, h, x, y, stage = node.g, node.h, node.x, node.y, node.stage
    h2g, collision, fuel, in_world = node.h2g, node.collision, node.fuel, node.in_world
    if mode == 0:
        heapq.heappush(open_list, (- stage, g + h, h, x, y, node))
    elif mode == 1:
        heapq.heappush(open_list, (g + h, h, x, y, node))
    elif mode == 2:
        heapq.heappush(open_list, (g + h, h, h2g, fuel, x, y, node))
    elif mode == 3:
        heapq.heappush(open_list, (g + h, h2g, h, fuel, x, y, node))
    elif mode == 4:
        heapq.heappush(open_list, (g + h, collision, h, h2g, fuel, x, y, gen, node))
    elif mode == 5:
        heapq.heappush(open_list, (g + h, collision, h, h2g, in_world, x, y, gen, node))
    else:
        raise NotImplementedError

def a_star(env, goal_info, constraints, aid, arg, earliest=0, latest=float('inf'), paths=None, block_actions=None):
    """
    A* search for a single-agent path in 3D space: finish a goal + return to border
    Node stages:
        0: move to border and interact with depo (if needed)
        1: finish goal
        2: move to border
    """
    goal = goal_info['goals'][aid]
    add, gx, gy, lv, _ = goal
    ax, ay = goal_info['pos'][aid]
    stage = goal_info['stage'][aid]
    cons, max_con_step = filter_constraints(constraints, aid)
    heu, teleport = arg.heu, arg.teleport

    # If the goal cannot be performed after all height changes, update time limit based on the last change
    # if not path2border(env, heights[-1], gx, gy):
    #     limit = min(limit, max(t2hid.keys()))

    # Get the height maps after the goal is finished
    if paths is not None and arg.order >= 4:
        if goal[0] == -1:
            ignore_goals = None
        else:
            ignore_goals = goal_info['all_succ'][goal].copy()
            ignore_goals.add(goal)
        heights, t2hid = construct_heights(env.height, block_actions, ignore_goals=ignore_goals)
        heights = np.tile(heights, (2, 1, 1, 1))
        heights[1, :, gx, gy] += 1 if add else -1
    else:
        heights, t2hid, ignore_goals = None, None, None

    open_list = []
    closed_list = dict()
    gen = 0
    h_val, h2g = heuristic(env, goal_info, stage, ax, ay, gx, gy, lv, heu, teleport)
    root = Node(None, 0, h_val, h2g, ax, ay, stage)
    push_node(open_list, root, gen, arg.order)

    while len(open_list) > 0:
        node = heapq.heappop(open_list)[-1]
        x, y, stage, g = node.x, node.y, node.stage, node.g

        '''Completion check: at stage 2, at parking location, no further constraints'''
        if stage == 2 and g >= max_con_step:
            if (teleport and x == -1) or (not teleport and (x, y) in env.border_loc):
                return get_path(node)

        '''Expand node'''
        children = []

        '''Stage 0: try to interact with depo'''
        if stage == 0:
            if arg.teleport and x == -1:
                h_val, h2g = heuristic(env, goal_info, 1, x, y, gx, gy, lv, heu, teleport)
                child = Node(node, g + 1, h_val, h2g, x, y, 1, action='depo')
                children.append(child)
            elif not arg.teleport and (x, y) in env.border_loc and not constrained(x, y, x, y, g + 1, 'depo', cons):
                h_val, h2g = heuristic(env, goal_info, 1, x, y, gx, gy, lv, heu, teleport)
                child = Node(node, g + 1, h_val, h2g, x, y, 1, action='depo')
                children.append(child)

        '''Stage 1: try to finish goal'''
        if g + 1 >= earliest and stage == 1 and (x, y) in goal_info['g2neighbor'][(gx, gy, lv)] \
                and not constrained(x, y, x, y, g + 1, 'goal', cons):
            h_val, h2g = heuristic(env, goal_info, 2, x, y, gx, gy, lv, heu, teleport)
            child = Node(node, g + 1, h_val, h2g, x, y, 2, action='goal')
            children.append(child)

        '''Move & stay actions'''
        for (x2, y2) in goal_info['movable'][(x, y)]:
            if constrained(x, y, x2, y2, g + 1, 'move', cons):
                continue
            h_val, h2g = heuristic(env, goal_info, stage, x2, y2, gx, gy, lv, heu, teleport)
            child = Node(node, g + 1, h_val, h2g, x2, y2, stage)
            children.append(child)

        '''Teleport: leave world & enter world'''
        if arg.teleport:
            if x == -1 or (x, y) in env.border_loc:
                h_val, h2g = heuristic(env, goal_info, stage, -1, -1, gx, gy, lv, heu, teleport)
                child = Node(node, g + 1, h_val, h2g, -1, -1, stage)
                children.append(child)
            if x == -1:
                for (x2, y2) in env.border_loc:
                    if constrained(x, y, x2, y2, g + 1, 'move', cons):
                        continue
                    h_val, h2g = heuristic(env, goal_info, stage, x2, y2, gx, gy, lv, heu, teleport)
                    child = Node(node, g + 1, h_val, h2g, x2, y2, stage)
                    children.append(child)

        '''Push children'''
        for child in children:
            if child.g <= latest:  # Trim nodes with g > limit
                if paths is not None and arg.order >= 4:
                    child.collision += count_collisions(child, aid, paths, heights, t2hid, block_actions, ignore_goals)
                key = (child.stage, child.x, child.y, child.g)
                val = (child.collision, child.in_world if arg.order == 5 else child.fuel)
                # New node, fewer collisions, or less fuel
                if key not in closed_list or closed_list[key] > val:
                    gen += 1
                    push_node(open_list, child, gen, arg.order)
                    closed_list[key] = val
    return None, None

def get_path(node):
    """
    Get a successful single-agent path to the goal, plus the time that goal is finished
    Result node in path: ((x, y), action)
    """
    path = []
    t, seen = 0, False
    while node:
        path.append(((node.x, node.y), node.action))
        if node.action == 'goal':
            seen = True
        if not seen:
            t += 1
        node = node.parent
    path.reverse()
    t = len(path) - t - 1
    return path, t

class Node:
    def __init__(self, parent, g, h, h2g, x, y, stage, action='move'):
        self.parent = parent
        self.g = g
        self.h = h
        self.h2g = h2g
        self.x = x
        self.y = y
        self.stage = stage
        self.action = action

        if parent:
            self.collision = parent.collision
            # Fuel consumption = 1 for each non-stay action
            self.fuel = parent.fuel
            if x != parent.x or y != parent.y or action != 'move':
                self.fuel += 1
            # In-world action = action starting from in-world location
            self.in_world = parent.in_world
            if x != -1:
                self.in_world += 1
        else:
            self.collision = 0
            self.fuel = 0
            self.in_world = 0
