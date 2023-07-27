"""Super class for centralized planning agents"""
import heapq
from threading import Lock

import numpy as np


class MetaAgent:
    def __init__(self, env, arg):
        self.fresh = False
        self.mutex = Lock()

        '''environment'''
        self.env = env
        self.task = arg.task
        self.agent_num = arg.num
        self.carry_mode = arg.carry_mode
        self.direction = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.subtask = arg.one_goal

        '''training'''
        self.epi_len = arg.episode_len

        '''algo'''
        self.algo = arg.algo

        '''visualization'''
        self.gif = arg.gif
        self.frame_path = arg.frame_path

        '''variables'''
        self.needs_replan = True
        self.goal = np.zeros((self.agent_num, 4), dtype=int)  # Goal: type (0 = finished, 1 = block, 2 = move), z, x, y

    '''Interaction'''
    def reset(self):
        # self.mutex.acquire()
        if not self.fresh:
            self.goal = np.zeros((self.agent_num, 4), dtype=int)
            self.fresh = True
            self.needs_replan = False
        # self.mutex.release()

    def update_goal(self, aid, ob):
        """
        Update an agent's goal, convert from 3D observation to [type, z, x, y]
        Args:
            ob: 3D goal channel from agent observation
            aid: agent id, 0 indexing
        """
        if ob.sum() == 0:  # Goal has been finished
            self.goal[aid] *= 0
        else:
            goal_type = ob.max()
            loc = np.transpose(np.nonzero(ob))
            self.goal[aid, 0] = goal_type
            self.goal[aid, 1:] = loc

    def observe(self):
        """Observe information from env: current agent position, carry status, height map"""
        self.agent_pos = self.env.agent_pos.copy()
        self.carry_status = [self.env._carry_block(self.agent_pos[i]) for i in range(self.agent_num)]
        self.height_map = self.env.height_map.copy()

    def process_goal(self):
        """
        Process goal block information
        If two non-source goals conflict, cancel one of them
        """
        self.goal_locs = []
        self.delta_hs = []
        for i in range(self.agent_num):
            if self.goal[i, 0] == 0:
                self.goal_locs.append((-1, -1))
                self.delta_hs.append(0)
                continue
            loc = tuple(self.goal[i, 2:])
            if [0, loc[0], loc[1]] in self.env.source:
                delta_h = 0
            elif self.carry_status[i]:
                delta_h = 1
            else:
                delta_h = -1
            self.goal_locs.append(loc)
            self.delta_hs.append(delta_h)
        for i in range(self.agent_num):
            for j in range(i):
                if self.goal_locs[i] != (-1, -1)\
                        and [0, self.goal_locs[i][0], self.goal_locs[i][1]] not in self.env.source\
                        and self.goal_locs[i] == self.goal_locs[j]:
                    di = self.manhattan(self.agent_pos[i], self.goal_locs[i])
                    dj = self.manhattan(self.agent_pos[j], self.goal_locs[j])
                    idx = i if di >= dj else j
                    self.goal[idx] *= 0
                    self.goal_locs[idx] = (-1, -1)
                    self.delta_hs[idx] = 0

    def plan(self):
        self.fresh = False
        self.observe()
        self.process_goal()
        self.paths = []
        # self.h_values = []
        # for i in range(self.agent_num):
        #     self.h_values.append(self.dijkstra(i))

    '''Env validation'''
    def move(self, loc, direction):
        return loc[0] + self.direction[direction][0], loc[1] + self.direction[direction][1]

    def movable(self, loc1, loc2, height_map, delta1=0, delta2=0):
        """Check if it is able to move from loc to loc2"""
        # loc2 should be within world
        if not self.env.in_world_2d(loc2):
            return False
        # Cannot move to the highest level
        if height_map[loc2[0], loc2[1]] + delta2 == self.env.h - 1:
            return False
        # Cannot move into source location
        if [0, loc2[0], loc2[1]] in self.env.source:
            return False
        # loc2 should be reachable from loc1 (height difference <= 1)
        if abs(height_map[loc1[0], loc1[1]] + delta1 - height_map[loc2[0], loc2[1]] - delta2) > 1:
            return False
        # Stay: height change should be the same
        if loc1 == loc2 and delta1 != delta2:
            return False
        return True

    def next_to_goal(self, loc, goal_loc):
        """
        Check if agent loc is next to goal loc
        Args:
            loc: agent location, (x, y)
            goal_loc: block goal location, (x, y)
        """
        return abs(loc[0] - goal_loc[0]) + abs(loc[1] - goal_loc[1]) == 1

    def manhattan(self, loc1, loc2):
        """Return the manhattan distance between a location to goal location"""
        if loc1[0] == -1 or loc2[0] == -1:
            return 0
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def block_ready(self, loc, goal_loc, carry, height_map):
        """
        Check if it's valid to finish the block goal
        Args:
            loc: agent location, (x, y)
            goal_loc: block goal location, (x, y)
            carry: agent carry status
            height_map: latest height map
        """
        # Agent should be next to goal loc
        if not self.next_to_goal(loc, goal_loc):
            return False
        # Already ready if goal is a source
        if [0, goal_loc[0], goal_loc[1]] in self.env.source:
            return True
        # Relative height should be correct
        h_a = height_map[loc[0], loc[1]]
        h_g = height_map[goal_loc[0], goal_loc[1]]
        return h_g - h_a == 1 - carry

    '''Search algo'''
    def dijkstra(self, aid):
        """
        Compute the shortest distance from all reachable locations to an agent's goal location, using Dijkstra
        Args:
            aid: agent id, 0 indexing
        """
        open_list = []  # (distance, location)
        heuristic = dict()  # location: distance
        start = tuple(self.goal[aid, 2:])
        heapq.heappush(open_list, (0, start))
        heuristic[start] = 0
        while len(open_list) > 0:
            (d, loc) = heapq.heappop(open_list)
            for direct in range(5):
                neighbor = self.move(loc, direct)
                neighbor_d = d + 1
                if not self.movable(loc, neighbor, self.height_map):  # Check if it's valid to move
                    continue
                if neighbor in heuristic:
                    existing_d = heuristic[neighbor]
                    if existing_d > neighbor_d:
                        heuristic[neighbor] = neighbor_d
                        heapq.heappush(open_list, (neighbor_d, neighbor))
                else:
                    heuristic[neighbor] = neighbor_d
                    heapq.heappush(open_list, (neighbor_d, neighbor))
        return heuristic

    def heu_manhattan(self, loc, goal_loc, block_goal=True):
        """Calculate h values from loc to goal_loc based on Manhattan distance"""
        h = self.manhattan(loc, goal_loc)
        # If loc is goal_loc, increase h values by 2 for block goal
        if block_goal and loc == goal_loc:
            h += 2
        return h

    def push_node(self, open_list, node):
        """
        Push a node to the open list
        Increasing order by: -1 * done * time, g + h, h, loc, node
        """
        heapq.heappush(open_list, (- node['done'] * node['time'], node['g'] + node['h'], node['h'], node['loc'], node))


    def filter_constraints(self, aid, constraints):
        """
        Filter out constraints for an agent
        Args:
            aid: agent id, 0 indexing
            constraints: all constraints in a CT node
        Returns:
            table: a constraint table for the agent, indexed by time
            max_constraint_step: maximum constraint time
        """
        table = dict()
        max_constraint_step = 0
        for con in constraints:
            if con['agent'] == aid:
                time = con['time']
                if time not in table.keys():
                    table[time] = [con]
                else:
                    if con not in table[time]:
                        table[time].append(con)
                max_constraint_step = max(max_constraint_step, time)
        return table, max_constraint_step

    def constrained(self, loc1, loc2, t, constraints, block):
        """
        Check if the action violates any constraints
        Args:
            loc1:
            loc2:
            t:
            constraints: all constraints related to current agent
            block: true for block action, false for move action
        """
        if t in constraints.keys():
            for con in constraints[t]:
                # Check vertex conflict
                if con['type'] == 'vertex' and loc2 == con['loc']:
                    return True
                # Block: check block conflict
                if block and con['type'] == 'block':
                    return True
                # Move: check edge conflict
                if not block and con['type'] == 'edge' and loc1 == con['loc'][0] and loc2 == con['loc'][1]:
                    return True
        return False

    def get_latest_map(self, t, maps, times, done, loc, delta_h):
        i = 0
        while i < len(times) - 1 and t >= times[i + 1]:
            i += 1
        h_map = maps[times[i]]
        if done:
            h_map = h_map.copy()
            h_map[loc[0], loc[1]] += delta_h
        return h_map

    def get_path(self, node):
        """Return a successful path to complete the goal, and info of block action time, location, and height change"""
        path = []
        block_action = None
        while node is not None:
            path.append((node['loc'], node['block']))
            if node['block'] is not None:
                block_action = path[-1]
            node = node['parent']
        path.reverse()
        if block_action is None:
            t = 0
            loc = (0, 0)
            delta_h = 0
        else:
            t = path.index(block_action)
            loc = block_action[1][0]
            delta_h = block_action[1][1]
        return path, (t, loc, delta_h)

    def a_star(self, aid, constraints, maps, times, goal_terminate=False, init_run=False):
        """
        Use A* search to find the shortest path for the agent to finish its goal and avoid collisions afterwards
        Notes:
            Goal type 0 (finished): avoid collision
            Goal type 1 (block): path to a valid location next to goal + block action
            Goal type 2 (move): path to goal location
            g value is only incremented when goal is not finished
        Node:
            done: goal completion
            block: (loc, delta_h) when performing a block action, none otherwise
        Args:
            aid: agent id, 0 indexing, agent being re-planned
            constraints: constraints by timestep
            maps: series of height map indexed by time
            times: timestamps of height map series
            goal_terminate: return path immediately upon goal completion, do not consider possible conflict in future
            init_run: initial run at the beginning of CBS, consider all possible height changes, can pick the 'best'
        Returns:
            The shortest path on success, None on failure
        """
        # Agent, goal, heuristics info
        goal_type = self.goal[aid, 0]
        carry = self.carry_status[aid]
        goal_loc = self.goal_locs[aid]
        delta_h = self.delta_hs[aid]
        start_loc = tuple(self.agent_pos[aid][1:])
        # h_values = self.h_values[aid]

        # Get constraints
        constraint_table, max_conflict_step = self.filter_constraints(aid, constraints)

        # Check initial vertex conflict
        if 0 in constraint_table.keys():
            for con in constraint_table[0]:
                if con['type'] == 'vertex' and con['loc'] == start_loc:
                    return None

        open_list = []
        closed_list = dict()  # (loc, time) : node
        root = {'g': 0, 'h': self.heu_manhattan(start_loc, goal_loc), 'loc': start_loc, 'time': 0, 'parent': None,
                'done': False, 'block': None, 'delta': 0}
        self.push_node(open_list, root)
        closed_list[(False, start_loc, 0)] = root

        while len(open_list) > 0:
            node = heapq.heappop(open_list)[-1]

            '''Completion check: finish goal (goal terminate) & avoid all conflicts (not goal terminate)'''
            if node['done'] and (goal_terminate or node['time'] >= max_conflict_step):
                return self.get_path(node)

            '''Generate child nodes: move / noop / block'''
            loc, done = node['loc'], node['done']
            children = []
            h_map = self.get_latest_map(node['time'] + 1, maps, times, done, goal_loc, delta_h)

            # Block: check if it's ready to perform block action
            if not done and goal_type == 1 and self.block_ready(loc, goal_loc, carry, h_map) \
               and not self.constrained(loc, loc, node['time'] + 1, constraint_table, block=True):
                child = {'g': node['g'] + 1, 'h': 0, 'loc': loc, 'time': node['time'] + 1, 'parent': node,
                         'done': True, 'block': (goal_loc, delta_h), 'delta': 0}
                children.append(child)

            # Move / noop
            for d in range(5):
                child_loc = self.move(loc, d)
                # Check if it's valid to move
                if not self.movable(loc, child_loc, h_map, node['delta']):
                    continue
                # Check move constraints
                if self.constrained(loc, child_loc, node['time'] + 1, constraint_table, block=False):
                    continue
                # Check goal completion for child node
                if goal_type == 0:
                    child_done = True
                elif goal_type == 1:
                    child_done = done
                else:
                    child_done = done or child_loc == goal_loc
                h_val = self.heu_manhattan(child_loc, goal_loc)
                child = {'g': node['g'] + 1, 'h': h_val, 'loc': child_loc, 'time': node['time'] + 1,
                         'parent': node, 'done': child_done, 'block': None, 'delta': 0}
                children.append(child)

            # init_run: do it again for possible height change
            if init_run:
                for d in range(5):
                    child_loc = self.move(loc, d)
                    if child_loc in self.goal_locs:
                        idx = self.goal_locs.index(child_loc)
                        if idx == aid or self.delta_hs[idx] == 0:  # Skip height change caused by own goal or no change
                            continue
                        if not self.movable(loc, child_loc, h_map, node['delta'], self.delta_hs[idx]):
                            continue
                        if goal_type == 0:
                            child_done = True
                        elif goal_type == 1:
                            child_done = done
                        else:
                            child_done = done or child_loc == goal_loc
                        h_val = self.heu_manhattan(child_loc, goal_loc)
                        child = {'g': node['g'] + 1, 'h': h_val, 'loc': child_loc, 'time': node['time'] + 1,
                                 'parent': node, 'done': child_done, 'block': None, 'delta': self.delta_hs[idx]}
                        children.append(child)

            for child in children:
                key = (child['done'], child['loc'], child['time'])
                if key not in closed_list.keys() and child['time'] <= self.epi_len:
                    closed_list[key] = child
                    self.push_node(open_list, child)

    def get_actions(self):
        actions = []
        for i in range(self.agent_num):
            node1, node2 = self.paths[i].pop(0), self.paths[i][0]
            if node2[1] is None:  # Move
                loc1, loc2 = node1[0], node2[0]
                for a, d in enumerate(self.direction):
                    if loc1[0] + d[0] == loc2[0] and loc1[1] + d[1] == loc2[1]:
                        actions.append(a)
                        break
            else:  # Block
                loc1, loc2 = node1[0], node2[1][0]
                for a, d in enumerate(self.direction):
                    if loc1[0] + d[0] == loc2[0] and loc1[1] + d[1] == loc2[1]:
                        if self.carry_status[i]:  # place
                            actions.append(a + 8)
                        else:  # pick
                            actions.append(a + 4)
                        break
                # self.needs_replan = True
        if len(actions) != self.agent_num:
            raise Exception('Invalid path')
        return actions
