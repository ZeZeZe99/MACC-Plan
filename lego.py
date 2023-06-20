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
        self.set_shadow(val=False)

        # Dynamic heuristic
        self.required_scaf_loc = set()
        self.unlocked_loc = set()

        # Learning
        self.cost = arg.cost
        self.R = arg.R
        self.T = arg.T
        self.translate = arg.translate

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
        if self.map == -1:
            self.goal = self.random_goal()
        else:
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
        ratio = correct / self.goal_total
        return {'total': self.goal_total, 'correct': correct, 'todo': todo, 'scaffold': scaffold, 'ratio': ratio,
                'stage': self.stage}

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

    '''=====Learning====='''
    def random_goal(self):
        while True:
            goal = np.random.randint(0, self.h, size=self.world_shape, dtype=np.int32)
            goal *= (1 - self.border)
            if (goal > 0).any():
                break
        return goal

    def set_radiation(self):
        radiation = self.goal3d.copy()
        for h in range(self.h):
            if radiation[0].sum() == 0:
                break
            change = True
            while change:
                change = False
                for x in range(1, self.w - 1):
                    for y in range(1, self.w - 1):
                        if self.goal3d[h, x, y] == 0:
                            continue
                        count = 0
                        val = radiation[h, x, y]
                        for (x2, y2) in self.valid_neighbor[(x, y)]:
                            if radiation[h, x2, y2] >= val:
                                count += 1
                            else:
                                break
                        if count == 4:
                            radiation[h, x, y] += 1
                            change = True
        self.radiation = radiation

    def set_bounding_box(self):
        """Find the minimal bounding box of the shadow region (2D)"""
        min_x, min_y = 0, 0
        max_x, max_y = self.w - 1, self.w - 1
        while (self.shadow[0, min_x, :] == 0).all():
            min_x += 1
        while (self.shadow[0, max_x, :] == 0).all():
            max_x -= 1
        while (self.shadow[0, :, min_y] == 0).all():
            min_y += 1
        while (self.shadow[0, :, max_y] == 0).all():
            max_y -= 1
        self.box = (min_x, max_x, min_y, max_y)

    def reset(self):
        self.set_goal()
        self.set_shadow()
        if self.R > 0:
            self.set_radiation()
        if self.translate:
            self.set_bounding_box()
        self.height = np.zeros(self.world_shape, dtype=np.int32)
        self.stage = 0
        return self.observe()

    def translate_ob(self, ob):
        """Translate the minimum bounding box of shadow region to (0, 0) corner"""
        ob = ob[self.box[0]:self.box[1] + 1, self.box[2]:self.box[3] + 1]
        ob = np.pad(ob, ((0, self.w - ob.shape[0]), (0, self.w - ob.shape[1])), 'constant')
        return ob

    def observe(self):
        if self.translate:
            return np.stack([self.translate_ob(self.height), self.translate_ob(self.goal)], axis=0)
        return np.stack([self.height, self.goal], axis=0)

    def built(self):
        return (self.height >= self.goal).all()

    def done(self):
        return (self.height == self.goal).all()

    def trap(self, x, y, add):
        """
        Check how many neighbors of (x, y) are trapped
        A location is trapped if:
            1. It's an unfinished goal location, and its 4 neighbors are all higher than it
            2. It's a scaffold location, and its 4 neighbors are all higher than it or equal to it
        """
        height = self.height.copy()
        if not add:  # Get the height before removing
            height[x, y] += 1
        trap = 0
        for (x2, y2) in self.valid_neighbor[(x, y)]:
            count = 0
            if height[x2, y2] < self.goal[x2, y2]:
                for (x3, y3) in self.valid_neighbor[(x2, y2)]:
                    if height[x3, y3] > height[x2, y2]:
                        count += 1
                    else:
                        break
            elif self.height[x2, y2] > self.goal[x2, y2]:
                for (x3, y3) in self.valid_neighbor[(x2, y2)]:
                    if height[x3, y3] >= height[x2, y2]:
                        count += 1
                    else:
                        break
            if count == 4:
                trap += 1
        return trap

    def reward(self, add, x, y):
        h = self.height[x, y] + (not add)  # height of the changed block
        scaffold = h > self.goal[x, y]

        reward = -self.cost

        if not scaffold:  # Goal block
            if self.R > 0:
                r = self.radiation[h, x, y]
            else:
                r = 1
            reward += r if add else -r  # reward adding, penalize removing
        elif self.stage == 1:  # Only consider scaffold in stage 1 (cleaning)
            r = 1
            reward += -r if add else r  # penalize adding, reward removing

        if self.T > 0:
            r = self.trap(x, y, add) * self.T
            reward += -r if add else r  # penalize trapping, reward un-trapping

        return reward

    def step(self, action):
        if action == -1:  # No valid action
            return self.observe(), -self.cost, self.done()
        remove, a = divmod(action, self.w * self.w)
        add = 1 - remove
        x, y = divmod(a, self.w)
        '''Validate, execute, get reward'''
        if add and self.height[x, y] + 1 < self.h:
            self.height[x, y] += 1
            reward = self.reward(add, x, y)
        elif not add and self.height[x, y] - 1 >= 0:
            self.height[x, y] -= 1
            reward = self.reward(add, x, y)
        else:
            reward = -self.cost
        self.stage = max(self.stage, int(self.built()))  # Enter stage 1 if built (never go back)
        return self.observe(), reward, self.done()
