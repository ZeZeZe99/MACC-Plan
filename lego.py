from collections import deque
import numpy as np
import goal

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
        self.height = np.zeros(self.world_shape, dtype=np.int32)
        self.world_shape3d = (self.h, self.w, self.w)
        self._set_world()

        # Plan
        self.map = arg.map
        if self.w == 5:
            self.goal_maps = np.clip(goal.GOAL_MAPS_5, 0, self.h - 1)
        elif self.w == 8:
            self.goal_maps = np.clip(goal.GOAL_MAPS_8, 0, self.h - 1)
        else:
            raise NotImplementedError
        self.set_goal()
        self.set_shadow()

        # Heuristic
        self.set_distance_map()
        self.set_support_map()

    '''Initialization'''
    def _set_world(self):
        """Set important properties of the world"""
        # Valid locations
        self.valid_loc = set()
        for i in range(self.w):
            for j in range(self.w):
                self.valid_loc.add((i, j))

        # Border: cannot place blocks
        self.border = np.zeros(self.world_shape, dtype=np.int32)
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
            self.valid_neighbor[(x, y)] = set()
            for (x2, y2) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if (x2, y2) in self.valid_loc:
                    self.valid_neighbor[(x, y)].add((x2, y2))

        # Search neighbors: no need to search border
        self.search_neighbor = dict()
        for (x, y) in self.valid_loc:
            if (x, y) not in self.border_loc:
                self.search_neighbor[(x, y)] = set()
                for (x2, y2) in self.valid_neighbor[(x, y)]:
                    if (x2, y2) not in self.border_loc:
                        self.search_neighbor[(x, y)].add((x2, y2))

        # Valid next locations (valid neighbor + current location)
        self.valid_next_loc = self.valid_neighbor.copy()
        for (x, y) in self.valid_neighbor:
            self.valid_next_loc[(x, y)].add((x, y))

    '''Goal related'''
    def set_goal(self):
        """Set goal map (2D)"""
        if self.map == -1:
            self.goal = self.random_goal()
        else:
            self.goal = np.array(self.goal_maps[self.map], dtype=np.int32)
        self.goal_total = self.goal.sum()
        self.goal3d = np.zeros(self.world_shape3d, dtype=np.int32)
        self.H = np.amax(self.goal)
        for lv in range(self.H):
            self.goal3d[lv] = self.goal > lv

    def random_goal(self):
        while True:
            goal = np.random.randint(0, self.h, size=self.world_shape, dtype=np.int32)
            goal *= (1 - self.border)
            if (goal > 0).any():
                break
        return goal

    def set_shadow(self):
        """
        Find the shadow region of the goal map (3D)
        """
        self.shadow = np.zeros(self.world_shape3d, dtype=np.int32)
        for x in range(1, self.w - 1):  # Skip border locations
            for y in range(1, self.w - 1):
                self.cast_shadow(self.goal[x, y] - 1, x, y)
        '''Filter only scaffold blocks'''
        self.scaf = self.shadow * (1 - self.goal3d)

    def cast_shadow(self, lv, x, y):
        if lv < 0:
            return
        if self.shadow[lv, x, y] == 1:
            return
        self.shadow[lv, x, y] = 1
        for (x2, y2) in [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (x2, y2) in self.valid_loc and (x2, y2) not in self.border_loc:
                self.cast_shadow(lv-1, x2, y2)

    '''Validation'''
    def valid_bfs_map(self, height, degree):
        reachable = self.border.copy()
        addable = np.zeros(self.world_shape, dtype=np.int32)
        removable = np.zeros(self.world_shape, dtype=np.int32)

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
                    if degree == 0 or self.shadow[h2, x2, y2] == 1:
                        addable[x2, y2] = 1
                    if degree == 0 or self.shadow[h, x, y] == 1:
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
                if degree == 0 or self.shadow[h, x, y] == 1:
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
                    if degree == 0 or self.shadow[h, x, y] == 1:
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

    '''Execution'''
    def execute(self, height, loc, add):
        x, y = loc
        if add:
            height[x, y] += 1
        else:
            height[x, y] -= 1
        return height

    '''Scaffold estimate'''
    def set_distance_map(self):
        """Get locations d distance away from the center (2D), d = 1, 2, ..., H-1"""
        maps = []
        for d in range(1, self.H):
            d_map = np.zeros((2*d+1, 2*d+1), dtype=np.int32)
            i = j = d
            for dx in range(-d, d + 1):
                x = i + dx
                dy = d - abs(dx)
                d_map[x, j + dy] = 1
                d_map[x, j - dy] = 1
            if d <= 2:
                maps.append(d_map)
            else:
                d_map[2:-2, 2:-2] = maps[-2]
                maps.append(d_map)
        self.distance_map = maps

    def find_d_support(self, x, y, d):
        """
        Find the d-support set of the goal at (x, y) (d distance away, d levels below)
        Equivalently, find the d-support goal of the scaffold at (x, y) (d distance away, d levels above)
        """
        d_map = self.distance_map[d - 1]
        s_map = np.zeros((self.w, self.w), dtype=np.int32)
        top_s, top_d = max(1, x - d), max(d - x + 1, 0)
        bottom_s, bottom_d = min(self.w - 1, x + d + 1), d + 1 + min(d, self.w - x - 2)
        left_s, left_d = max(1, y - d), max(d - y + 1, 0)
        right_s, right_d = min(self.w - 1, y + d + 1), d + 1 + min(d, self.w - y - 2)
        s_map[top_s:bottom_s, left_s:right_s] = d_map[top_d:bottom_d, left_d:right_d]
        return s_map

    def set_support_map(self):
        """Get the d-support set of each goal"""
        goal2support = dict()
        for lv in range(1, self.H):
            for (x, y) in np.transpose(np.nonzero(self.goal3d[lv])):
                goal2support[(lv, x, y)] = np.stack([self.find_d_support(x, y, d) for d in range(1, lv+1)])
        self.goal2support = goal2support

    def filter(self, lv, support, height):
        """
        Filter a support set out (may not need a scaffold in it) if it contains:
            1. a scaffold that's already added
            2.1. a goal
            2.2. a goal that doesn't have another already added goal on top
        Args:
            lv: check at level lv (height lv+1)
            support: support set at level lv (height lv+1)
            height: current height map
        """
        block_from_lv = np.clip(height - lv, 0, None)
        goal_from_lv = np.clip(self.goal - lv, 0, None)
        scaffold = np.clip(block_from_lv - goal_from_lv, 0, None)
        # Use 2.1
        # has_support = support * np.logical_or(scaffold > 0, goal_from_lv > 0)
        # Use 2.2
        no_goal_above = np.logical_and(goal_from_lv, np.logical_or(goal_from_lv == 1, block_from_lv <= 1))
        has_support = support * np.logical_or(scaffold > 0, no_goal_above)
        return (has_support > 0).any()

    def find_groups(self, height):
        """Find groups of unfinished goals"""
        groups = []
        lv2group = dict()
        for lv in range(1, self.H):
            workspace = self.goal3d[lv] * (height <= lv)  # Unfinished goals at level lv
            group_maps, groups_loc = [], []
            for x in range(1, self.w - 1):
                for y in range(1, self.w - 1):
                    if workspace[x, y] == 1:
                        g_map = np.zeros((self.w, self.w), dtype=np.int32)
                        group = []
                        self.connect_goals(workspace, x, y, g_map, group)
                        group_maps.append(g_map)
                        groups_loc.append(group)
            groups.append((group_maps, groups_loc))
            lv2group[lv] = len(group_maps)
        return groups, lv2group

    def connect_goals(self, workspace, x, y, group_map, group):
        """Connect neighboring goals into a group"""
        if workspace[x, y] == 0:
            return
        workspace[x, y] = 0
        group_map[x, y] = 1
        group.append((x, y))
        for (x2, y2) in self.search_neighbor[(x, y)]:
            self.connect_goals(workspace, x2, y2, group_map, group)

    def find_group_support(self, groups):
        """Find the d-support set of each group"""
        group2support = dict()
        for lv in range(1, self.H):
            group_maps, group_loc = groups[lv - 1]
            for i in range(len(group_maps)):
                s_map = np.zeros((lv, self.w, self.w), dtype=np.int32)
                for d in range(1, lv + 1):
                    for (x, y) in group_loc[i]:
                        s_map[d-1] += self.goal2support[(lv, x, y)][d-1]
                    s_map[d-1] = np.clip(s_map[d-1] * (1 - group_maps[i]), 0, 1)
                group2support[(lv, i)] = s_map
        return group2support

    def cast_scaffold_value_group(self, height, group2support, lv2group):
        """Cast scaffold value for each group to each level"""
        scaffold_v = np.zeros(self.world_shape3d, dtype=np.int32)
        useful_support = dict()
        for lv in range(1, self.H):
            for i in range(lv2group[lv]):
                useful_support[(lv, i)] = set()
                supports = group2support[(lv, i)]
                for d in range(1, lv + 1):
                    d_support = supports[d - 1]
                    if self.filter(lv - d, d_support, height):
                        continue
                    scaffold_v[lv - d] += d_support
                    useful_support[(lv, i)].add(d)
        return scaffold_v, useful_support

    def get_goal_val_group(self, height):
        """Calculate goal value for each group"""
        groups, lv2group = self.find_groups(height)
        group2support = self.find_group_support(groups)
        scaffold_v, useful_support = self.cast_scaffold_value_group(height, group2support, lv2group)
        goal_v = np.empty(self.H - 1, dtype=np.float32)
        for lv in range(1, self.H):
            val = 0
            for i in range(lv2group[lv]):
                if (lv, i) not in useful_support:
                    continue
                supports = group2support[(lv, i)]
                for d in useful_support[(lv, i)]:
                    val += 1 / (scaffold_v[lv - d] * supports[d - 1]).max()
            goal_v[lv - 1] = val
        return goal_v

    '''Other goal value estimates (worse version)'''
    def cast_scaffold_value(self, height):
        scaffold_v = np.zeros(self.world_shape3d, dtype=np.int32)
        useful_support = dict()
        for lv in range(1, self.H):
            for (x, y) in np.transpose(np.nonzero(self.goal3d[lv] * (height <= lv))):  # Unfinished goals at level lv
                useful_support[(lv, x, y)] = set()
                supports = self.goal2support[(lv, x, y)]
                for d in range(1, lv + 1):
                    d_support = supports[d - 1]
                    if self.filter(lv - d, d_support, height):
                        continue
                    scaffold_v[lv - d] += d_support
                    useful_support[(lv, x, y)].add(d)
        return scaffold_v, useful_support

    def get_goal_val(self, height):
        scaffold_v, useful_support = self.cast_scaffold_value(height)
        goal_v = np.zeros(self.world_shape3d, dtype=np.float32)
        for lv in range(1, self.H):
            for (x, y) in np.transpose(np.nonzero(self.goal3d[lv])):
                if (lv, x, y) not in useful_support:
                    continue
                supports = self.goal2support[(lv, x, y)]
                for d in useful_support[(lv, x, y)]:
                    goal_v[lv, x, y] += 1 / (scaffold_v[lv - d] * supports[d - 1]).max()
        return goal_v

    def get_goal_val_nb(self, height):
        vgs = np.zeros(self.world_shape3d, dtype=np.float32)
        for z in range(1, self.H):
            vg = self.goal3d[z].copy().astype(np.float32)
            for (x, y) in np.transpose(np.nonzero(vg)):
                if height[x, y] >= z:
                    vg[x, y] = 0
                    continue
                for (x2, y2) in self.search_neighbor[(x, y)]:
                    if height[x2, y2] >= z or self.goal[x2, y2] >= z:
                        vg[x, y] = 0
                        break
            scaf = 1 - self.goal3d[z-1] - self.border
            vc = np.zeros((self.w, self.w), dtype=np.int32)
            for (x, y) in np.transpose(np.nonzero(scaf)):
                for (x2, y2) in self.search_neighbor[(x, y)]:
                    if vg[x2, y2] > 0:
                        vc[x, y] += 1
            for (x, y) in np.transpose(np.nonzero(vg)):
                vcs = [vc[x2, y2] for (x2, y2) in self.search_neighbor[(x, y)]]
                vg[x, y] = 1 / max(vcs)
            vgs[z] = vg
        return vgs