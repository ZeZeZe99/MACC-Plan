from collections import deque
import numpy as np
import goal

class GridWorld:
    def __init__(self, arg):
        # Environment
        self.h = arg.h
        self.w = arg.w
        self.world_shape = (self.w, self.w)
        self.height = np.zeros(self.world_shape, dtype=np.int32)
        self.world_shape3d = (self.h, self.w, self.w)
        self.world = np.zeros(self.world_shape3d, dtype=np.int32)
        self._set_world()

        # Plan
        self.map = arg.map
        if self.w == 5:
            self.goal_maps = goal.GOAL_MAPS_5
        elif self.w == 8:
            self.goal_maps = goal.GOAL_MAPS_8
        else:
            raise NotImplementedError
        self.set_goal()
        self.set_shadow(val=True)

        # Dynamic heuristic
        self.required_scaf_loc = set()
        self.unlocked_loc = set()

        # Agent
        self.num = arg.num

    def _set_world(self):
        """Set important properties of the world"""
        '''Valid location'''
        self.valid_loc = set()
        for i in range(self.w):
            for j in range(self.w):
                self.valid_loc.add((i, j))
        self.valid_neighbor = dict()
        for (x, y) in self.valid_loc:
            self.valid_neighbor[(x, y)] = set()
            for (x2, y2) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if (x2, y2) in self.valid_loc:
                    self.valid_neighbor[(x, y)].add((x2, y2))

        '''Border: cannot place blocks'''
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

        '''Start location: locations next to border, for valid action checking'''
        self.start_loc = set()
        for i in range(1, self.w - 1):
            self.start_loc.add((1, i))
            self.start_loc.add((self.w - 2, i))
            self.start_loc.add((i, 1))
            self.start_loc.add((i, self.w - 2))

        '''Search neighbors: no need to search border'''
        self.search_neighbor = dict()
        for (x, y) in self.valid_loc:
            if (x, y) not in self.border_loc:
                self.search_neighbor[(x, y)] = set()
                for (x2, y2) in self.valid_neighbor[(x, y)]:
                    if (x2, y2) not in self.border_loc:
                        self.search_neighbor[(x, y)].add((x2, y2))

    '''Goal related'''
    def set_goal(self):
        """Set goal map (2D)"""
        self.goal = np.array(self.goal_maps[self.map], dtype=np.int32)
        self.goal_total = self.goal.sum()
        self.goal3d = np.zeros(self.world_shape3d, dtype=np.int32)
        for h in range(self.h):
            self.goal3d[h] = self.goal > h

    def set_shadow(self, val=False):
        """
        Find the shadow region of the goal map (3D)
        Args:
            val: True = calculate shadow value, False = only find shadow location
        """
        self.shadow = np.zeros(self.world_shape3d, dtype=np.int32)
        self.shadow_val = np.zeros(self.world_shape3d, dtype=np.int32)
        self.shadow_loc = set()
        for x in range(self.w):
            for y in range(self.w):
                '''Cast shadow'''
                # for z in range(1, self.goal[x, y]):
                #     self.cast_shadow(z, x, y, val)
                self.cast_shadow(self.goal[x, y] - 1, x, y, val)
        '''Filter only scaffold blocks'''
        self.scaf = self.shadow * (1 - self.goal3d)
        '''Make a dictionary of shadow values for scaffold blocks'''
        self.shadow_vald = dict()
        for (h, x, y) in self.shadow_loc:
            if self.scaf[h, x, y] == 1:
                self.shadow_vald[(h, x, y)] = self.shadow_val[h, x, y]

    def cast_shadow(self, h, x, y, val=False):
        if h < 0:
            return
        if not val and self.shadow[h, x, y] == 1:
            return
        elif val:
            self.shadow_val[h, x, y] += 1
        self.shadow[h, x, y] = 1
        self.shadow_loc.add((h, x, y))
        for (x2, y2) in [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (x2, y2) in self.valid_loc and (x2, y2) not in self.border_loc:
                self.cast_shadow(h-1, x2, y2, val)

    '''Validation'''
    def valid_action(self, height, mode=0, degree=0):
        """
        Find valid locations to reach, add / remove blocks
        Args:
            height: current height map
            mode: 0 = use BFS and set, 1 = use BFS and map
            degree: 0 = all valid locations, 1 = only in shadow region
        """
        if mode == 0:
            return self.valid_bfs_set(height, degree)
        elif mode == 1:
            return self.valid_bfs_map(height, degree)
        else:
            raise NotImplementedError

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

    def valid_bfs_set(self, height, degree):
        reachable = self.border_loc.copy()  # (x, y)
        action = set()  # (x, y, add)

        queue = deque()
        visited = set()
        for (x, y) in self.start_loc:  # Start from locations next to border
            if height[x, y] <= 1:  # Only consider locations reachable from border
                queue.append((x, y))
                reachable.add((x, y))

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
                    reachable.add((x2, y2))
                if h > h2:  # Remove
                    action.add((x, y, False))
                elif h < h2:  # Remove
                    action.add((x2, y2, False))
                else:  # Add
                    if degree == 0 or (h2, x2, y2) in self.shadow_loc:
                        action.add((x2, y2, True))
                    if degree == 0 or (h, x, y) in self.shadow_loc:
                        action.add((x, y, True))
        return reachable, action

    '''Status'''
    def status(self, height):
        correct = np.clip(height, 0, self.goal).sum()
        todo = self.goal_total - correct
        scaffold = np.clip(height - self.goal, 0, self.h).sum()
        return todo, scaffold

    '''Execution'''
    def execute(self, height, loc, add):
        x, y = loc
        h = height[x, y]
        if add:
            h += 1
        else:
            h -= 1
        height[x, y] = h
        return height
