from collections import deque
import numpy as np
import goal
from copy import deepcopy

"""
Variables:
    h = height
    lv = level (0 index)
    h = 1 <=> lv = 0
    env.h = max (goal structure height) + 1
"""

class GridWorld:
    def __init__(self, arg):
        # Environment
        self.h = arg.h
        self.w = arg.w
        self.world_shape = (self.w, self.w)
        self.height = np.zeros(self.world_shape, dtype=np.int8)
        self._set_world()

        # Plan
        self.map = arg.map
        if self.w == 10:
            self.goal_maps = np.clip(goal.GOAL_MAPS_10, 0, self.h - 1)
        elif self.w == 8:
            self.goal_maps = np.clip(goal.GOAL_MAPS_8, 0, self.h - 1)
        else:
            raise NotImplementedError

    '''Initialization'''
    def _set_world(self):
        """Set important properties of the world"""
        # Valid locations
        self.valid_loc = set()
        for i in range(self.w):
            for j in range(self.w):
                self.valid_loc.add((i, j))

        # Border: cannot place blocks
        self.border = np.zeros(self.world_shape, dtype=np.int8)
        self.border[0, :] = 1
        self.border[-1, :] = 1
        self.border[:, 0] = 1
        self.border[:, -1] = 1
        self.border_loc = set()
        for i in range(self.w):
            self.border_loc.add((0, i))
            self.border_loc.add((self.w - 1, i))
            self.border_loc.add((i, 0))
            self.border_loc.add((i, self.w - 1))

        # Start location: locations next to border, to speed up valid action checking
        self.start_loc = set()
        for i in range(1, self.w - 1):
            self.start_loc.add((1, i))
            self.start_loc.add((self.w - 2, i))
            self.start_loc.add((i, 1))
            self.start_loc.add((i, self.w - 2))

        # Valid neighbors
        self.valid_neighbor = dict()
        for (x, y) in self.valid_loc:
            self.valid_neighbor[(x, y)] = []
            for (x2, y2) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if (x2, y2) in self.valid_loc:
                    self.valid_neighbor[(x, y)].append((x2, y2))

        # Search neighbors: no need to search border
        self.search_neighbor = dict()
        for (x, y) in self.valid_loc:
            if (x, y) not in self.border_loc:
                self.search_neighbor[(x, y)] = []
                for (x2, y2) in self.valid_neighbor[(x, y)]:
                    if (x2, y2) not in self.border_loc:
                        self.search_neighbor[(x, y)].append((x2, y2))

        # Valid next locations (valid neighbor + current location)
        self.valid_next_loc = deepcopy(self.valid_neighbor)
        self.valid_next_loc[(-1, -1)] = []
        for (x, y) in self.valid_neighbor:
            self.valid_next_loc[(x, y)].append((x, y))

        # Direction
        self.dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    '''Goal related'''
    def set_goal(self):
        """Set goal map (2D)"""
        if self.map == -1:
            self.goal = self.random_goal()
        else:
            self.goal = np.array(self.goal_maps[self.map], dtype=np.int8)
        self.goal_total = self.goal.sum()
        self.H = np.amax(self.goal)
        self.world_shape3d = (self.H, *self.world_shape)
        self.goal3d = np.zeros(self.world_shape3d, dtype=np.int8)
        for lv in range(self.H):
            self.goal3d[lv] = self.goal > lv

    def random_goal(self):
        while True:
            goal = np.random.randint(0, self.h, size=self.world_shape, dtype=np.int8)
            goal *= (1 - self.border)
            if (goal > 0).any():
                break
        return goal

    def set_shadow(self, val=False):
        """
        Find the shadow region of the goal map (3D)
        Calculate shadow value when val=True: number of shadow regions each block belongs to
        """
        self.shadow = np.zeros(self.world_shape3d, dtype=np.int8)
        self.shadow_val = np.zeros(self.world_shape3d, dtype=np.int8)
        for x in range(1, self.w - 1):  # Skip border locations
            for y in range(1, self.w - 1):
                shadow = np.zeros(self.world_shape3d, dtype=np.int8) if val else self.shadow
                self.cast_shadow(self.goal[x, y] - 1, x, y, shadow)
                if val:
                    self.shadow |= shadow
                    self.shadow_val += shadow
        self.shadow_height = np.sum(self.shadow, axis=0)

        '''Filter only scaffold blocks'''
        self.scaf = self.shadow * (1 - self.goal3d)

    def cast_shadow(self, lv, x, y, shadow):
        if lv < 0:
            return
        if shadow[lv, x, y] == 1:
            return
        shadow[lv, x, y] = 1
        for (x2, y2) in [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (x2, y2) in self.valid_loc and (x2, y2) not in self.border_loc:
                self.cast_shadow(lv-1, x2, y2, shadow)

    '''Validation'''
    def valid_bfs_map(self, height, degree):
        reachable = self.border.copy()
        addable = np.zeros(self.world_shape, dtype=np.int8)
        removable = np.zeros(self.world_shape, dtype=np.int8)

        queue = deque()
        visited = set()
        for (x, y) in self.start_loc:  # Start from locations next to border
            if height[x, y] <= 1:  # Only consider locations reachable from border
                queue.append((x, y))
                reachable[x, y] = 1
                if height[x, y] == 1:
                    removable[x, y] = 1
                elif height[x, y] == 0 and (degree == 0 or self.shadow[0, x, y] == 1):
                    addable[x, y] = 1

        while len(queue) > 0:
            x, y = queue.popleft()
            visited.add((x, y))
            h = height[x, y]

            for (x2, y2) in self.search_neighbor[(x, y)]:  # Do not consider border
                if (x2, y2) in visited:
                    continue
                h2 = height[x2, y2]
                if abs(h - h2) > 1:
                    continue
                if (x2, y2) not in queue:
                    queue.append((x2, y2))
                    reachable[x2, y2] = 1
                if h > h2:  # Remove
                    removable[x, y] = 1
                elif h < h2:  # Remove
                    removable[x2, y2] = 1
                else:  # Add
                    if degree == 0 or (h < self.H and self.shadow[h, x2, y2] == 1):
                        addable[x2, y2] = 1
                    if degree == 0 or (h < self.H and self.shadow[h, x, y] == 1):
                        addable[x, y] = 1
        return np.stack([reachable, addable, removable], axis=0)

    def path_to_border(self, height, x, y):
        """Check if there is a valid path from (x, y) to border, with DFS"""
        queue = deque()
        visited = set()
        queue.append((x, y))
        while len(queue) > 0:
            x, y = queue.pop()
            if self.border[x, y] == 1:
                return True, None
            if (x, y) not in visited:
                visited.add((x, y))
                h = height[x, y]
                for (x2, y2) in self.valid_neighbor[(x, y)]:
                    if (x2, y2) not in queue and abs(h - height[x2, y2]) <= 1:
                        queue.append((x2, y2))
        # If no path found, return all visited locations
        return False, visited

    def update_block(self, height, x, y, valid_map, degree):
        """Update add / remove status at location (x, y)"""
        h = height[x, y]
        for (x2, y2) in self.valid_neighbor[(x, y)]:
            h2 = height[x2, y2]
            if h2 == h - 1:
                valid_map[2, x, y] = 1
            elif h2 == h:
                if degree == 0 or (h < self.H and self.shadow[h, x, y] == 1):
                    valid_map[1, x, y] = 1

    def update_valid_map(self, height, x, y, old_valid_map, degree):
        """Incrementally update valid map after adding / removing block at (x, y)"""
        if not self.path_to_border(height, x, y)[0]:  # No path back to border (invalid)
            return False, None
        valid_map = old_valid_map.copy()
        reach = deque()
        # Update add / remove of location (x, y) and its neighbors
        h = height[x, y]
        valid_map[1:, x, y] = 0
        for (x2, y2) in self.valid_neighbor[(x, y)]:
            h2 = height[x2, y2]
            if abs(h - h2) > 1:  # Neighbor not reachable from (x, y)
                if valid_map[0, x2, y2] == 1:  # Originally reachable, may become unreachable
                    path, locations = self.path_to_border(height, x2, y2)
                    if path:
                        valid_map[1:, x2, y2] = 0
                        self.update_block(height, x2, y2, valid_map, degree)
                    else:
                        for (x3, y3) in locations:
                            valid_map[:, x3, y3] = 0
            else:  # Neighbor reachable from (x, y)
                if h2 == h - 1:
                    valid_map[2, x, y] = 1
                elif h2 == h:
                    if degree == 0 or (h < self.H and self.shadow[h, x, y] == 1):
                        valid_map[1, x, y] = 1
                if valid_map[0, x2, y2] == 0:
                    reach.append((x2, y2))
                else:
                    valid_map[1:, x2, y2] = 0  # Originally unreachable, just become reachable
                    self.update_block(height, x2, y2, valid_map, degree)

        # Neighbors of (x, y) that become reachable
        visited = set()
        while len(reach) > 0:
            x2, y2 = reach.popleft()
            if (x2, y2) in visited:
                continue
            h2 = height[x2, y2]
            valid_map[0, x2, y2] = 1  # Newly reachable
            self.update_block(height, x2, y2, valid_map, degree)
            for (x3, y3) in self.valid_neighbor[(x2, y2)]:
                h3 = height[x3, y3]
                if abs(h2 - h3) <= 1 and valid_map[0, x3, y3] == 0 and (x3, y3) not in reach:
                    reach.append((x3, y3))
        return True, valid_map

    '''Used for scaffold estimate'''
    def set_distance_map(self):
        """Get locations d distance away from the center (2D), d = 1, 2, ..., H-1"""
        maps = []
        for d in range(1, self.H):
            d_map = np.zeros((2*d+1, 2*d+1), dtype=np.int8)
            i = j = d
            for dx in range(-d, d + 1):
                x = i + dx
                dy = d - abs(dx)
                d_map[x, j + dy] = 1
                d_map[x, j - dy] = 1
            if d <= 2:
                maps.append(d_map)
            else:
                d_map[2:-2, 2:-2] |= maps[-2]
                maps.append(d_map)
        self.distance_map = maps

    def set_support_map(self):
        """Set the d-support set of each goal, ordered by support level"""
        goal2support = dict()
        for lv in range(1, self.H):
            for (x, y) in np.transpose(np.nonzero(self.goal3d[lv])):
                supports = [self.find_d_support(x, y, d) for d in range(lv, 0, -1)]
                goal2support[(lv, x, y)] = np.stack(supports, axis=0)
        self.goal2support = goal2support

    def find_d_support(self, x, y, d):
        """
        Find the d-support set of the goal at (x, y) (d distance away, d levels below)
        Equivalently, find the d-support goal of the scaffold at (x, y) (d distance away, d levels above)
        """
        d_map = self.distance_map[d - 1]
        s_map = np.zeros((self.w, self.w), dtype=np.int8)
        top_s, top_d = max(1, x - d), max(d - x + 1, 0)
        bottom_s, bottom_d = min(self.w - 1, x + d + 1), d + 1 + min(d, self.w - x - 2)
        left_s, left_d = max(1, y - d), max(d - y + 1, 0)
        right_s, right_d = min(self.w - 1, y + d + 1), d + 1 + min(d, self.w - y - 2)
        s_map[top_s:bottom_s, left_s:right_s] = d_map[top_d:bottom_d, left_d:right_d]
        return s_map

    '''Symmetry detection'''
    def set_light(self):
        self.light_val = np.zeros(self.world_shape3d, dtype=np.int8)
        for x, y in np.transpose(np.nonzero(self.shadow_val[0] > 1)):
            if self.shadow_val[0, x, y] > self.shadow_val[1, x, y]:
                light = self.cast_light(x, y) * (self.shadow_val[0, x, y] - 1)
                self.light_val += light

    def cast_light(self, x, y):
        light = np.zeros(self.world_shape3d, dtype=np.int8)
        light[-1] = self.find_d_support(x, y, self.H - 1) * self.goal3d[-1]
        for z in range(self.H - 2, 0, -1):
            light[z] = light[z + 1]
            for sx, sy in np.transpose(np.nonzero(self.find_d_support(x, y, z))):
                if self.goal3d[z, sx, sy] == 1:
                    light[z, sx, sy] = 1
                    continue
                for nx, ny in self.search_neighbor[(sx, sy)]:
                    if light[z + 1, nx, ny] == 1:
                        light[z, sx, sy] = 1
                        break
        light[0] = light[1]
        light[0, x, y] = 1
        return light

    '''Low level search functions'''
    def set_mirror_map(self):
        """Create a series of mirror maps (original map with its flip over 4 borders)"""
        w = self.w
        mirror_map = np.zeros((3 * w, 3 * w), dtype=np.int8)
        mirror_map[:, w:2 * w] = 1
        mirror_map[w:2 * w, :] = 1
        self.mirror_neighbor = dict()
        for (x, y) in np.transpose(np.nonzero(mirror_map)):
            self.mirror_neighbor[(x, y)] = set()
            for (x2, y2) in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
                if x2 < 0 or x2 >= 3 * w or y2 < 0 or y2 >= 3 * w:
                    continue
                if mirror_map[x2, y2] == 0:
                    continue
                self.mirror_neighbor[(x, y)].add((x2, y2))
        self.origin2mirror = dict()
        self.mirror2origin = dict()
        for (x, y) in self.valid_loc:
            self.origin2mirror[(x, y)] = [(w - x - 1, y + w), (x + w, w - y - 1),
                                          (3 * w - x - 1, y + w), (x + w, 3 * w - y - 1)]
            self.mirror2origin[(w - x - 1, y + w)] = (x, y)
            self.mirror2origin[(x + w, w - y - 1)] = (x, y)
            self.mirror2origin[(3 * w - x - 1, y + w)] = (x, y)
            self.mirror2origin[(x + w, 3 * w - y - 1)] = (x, y)
            self.mirror2origin[(x + w, y + w)] = (x, y)
