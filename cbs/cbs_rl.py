import heapq
import time

from agents.meta_agent import MetaAgent

class CBS(MetaAgent):
    def __init__(self, env, arg):
        super().__init__(env, arg)
        self.timeout = 10

    def get_location(self, path, t):
        return path[t][0]

    def get_block_action(self, path, t):
        return path[t][1]

    def detect_conflict(self, path1, path2, block1, block2, maps, times, w):
        """
        Detect conflict between 2 paths up to w timesteps
        Notes:
            Assume 1-robust setting
            Agent cannot move to / perform block action at a location occupied by another agent in the previous step
        Args:
            path1:
            path2:
            block1: block action info
            block2:
            w: conflict consideration window
        Returns:
            The first conflict detected, or None if no conflict.
            Conflict format: {'type', 'loc', 'time'}
        """
        block_t1, block_loc1, delta_h1 = block1
        block_t2, block_loc2, delta_h2 = block2
        idx = 0
        height_map = maps[times[idx]]

        for t in range(1, min(w, len(path1), len(path2))):
            # Update height map
            if idx < len(times) - 1 and t == times[idx + 1]:
                idx += 1
                height_map = maps[t]

            loc1, loc2 = self.get_location(path1, t), self.get_location(path2, t)
            prev_loc1, prev_loc2 = self.get_location(path1, t - 1), self.get_location(path2, t - 1)

            # Check vertex conflict
            if loc1 == loc2:
                return {'type': 'vertex', 'loc': loc1, 'time': t}
            # Check 1-delay vertex conflict
            if loc1 == prev_loc2:
                return {'type': 'vertex', 'loc': loc1, 'time': t-1}
            if loc2 == prev_loc1:
                return {'type': 'vertex', 'loc': loc2, 'time': t - 1}
            # Check edge conflict
            if prev_loc1 == loc2 and prev_loc2 == loc1:
                return {'type': 'edge', 'loc': (prev_loc1, loc1), 'time': t}

            # Check agent-block conflict
            if t == block_t1:
                if block_loc1 == prev_loc2:  # Agent leaving block location
                    return {'type': 'agent-block', 'loc': block_loc1, 'time': t-1, 'block': 1, 'arrive': False}
                if block_loc1 == loc2:  # Agent arriving at block location
                    return {'type': 'agent-block', 'loc': block_loc1, 'time': t, 'block': 1, 'arrive': True}
            if t == block_t2:
                if block_loc2 == prev_loc1:
                    return {'type': 'agent-block', 'loc': block_loc2, 'time': t-1, 'block': 2, 'arrive': False}
                if block_loc2 == loc1:
                    return {'type': 'agent-block', 'loc': block_loc2, 'time': t, 'block': 2, 'arrive': True}
            # Check block-block conflict
            if t == block_t1 and t == block_t2:
                if block_loc1 == block_loc2 and [0, block_loc1[0], block_loc1[1]] not in self.env.source:
                    return {'type': 'block-block', 'loc': block_loc1, 'time': t}

            # Check map height conflict: cannot go to the highest level
            if height_map[loc1[0], loc1[1]] == self.env.h - 1 and loc1 == block_loc2:
                return {'type': 'height', 'loc': loc1, 'time': t, 'block': 2, 'block_time': block_t2}
            if height_map[loc2[0], loc2[1]] == self.env.h - 1 and loc2 == block_loc1:
                return {'type': 'height', 'loc': loc2, 'time': t, 'block': 1, 'block_time': block_t1}
            # Check map reach conflict: relative height <= 1
            if abs(height_map[loc1[0], loc1[1]] - height_map[prev_loc1[0], prev_loc1[1]]) > 1 \
                    and (loc1 == block_loc2 or prev_loc1 == block_loc2):
                return {'type': 'map', 'loc': (prev_loc1, loc1), 'time': t, 'block': 2, 'block_time': block_t2}
            if abs(height_map[loc2[0], loc2[1]] - height_map[prev_loc2[0], prev_loc2[1]]) > 1 \
                    and (loc2 == block_loc1 or prev_loc2 == block_loc1):
                return {'type': 'map', 'loc': (prev_loc2, loc2), 'time': t, 'block': 1, 'block_time': block_t1}

        return None

    def detect_all_conflicts(self, paths, infos, w):
        maps, times = self.construct_height_series(infos)
        conflicts = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                c = self.detect_conflict(paths[i], paths[j], infos[i], infos[j], maps, times, w=w)
                if c is None:
                    continue
                c['a1'] = i
                c['a2'] = j
                conflicts.append(c)
        return conflicts

    def resolve_conflict(self, conflict, w):
        """
        Resolve a conflict between two agents, create constraints for both agents
        Resolving 1-delay conflicts through symmetric range constraints
        Args:
            conflict: a conflict between two agents
            {'type': conflict type, 'a1' 'a2': agent index, 'loc': conflict location, 'time': conflict time,
             'block': agent performing block action, 'block_time': block action time}
             w: conflict consideration window
        Returns:
            List of 2 lists of constraints.
            Constraints: {'type': constraint type, 'agent': constraint agent, 'loc': constraint location,
                          'time': constraint time}
        """
        constraint1, constraint2 = [], []
        loc = conflict['loc']
        time = conflict['time']
        if conflict['type'] == 'vertex':
            constraint1.append({'type': 'vertex', 'agent': conflict['a1'], 'loc': loc, 'time': time})
            constraint1.append({'type': 'vertex', 'agent': conflict['a1'], 'loc': loc, 'time': time + 1})
            constraint2.append({'type': 'vertex', 'agent': conflict['a2'], 'loc': loc, 'time': time})
            constraint2.append({'type': 'vertex', 'agent': conflict['a2'], 'loc': loc, 'time': time + 1})
        elif conflict['type'] == 'edge':
            constraint1.append({'type': 'edge', 'agent': conflict['a1'], 'loc': loc, 'time': time})
            loc2 = (loc[1], loc[0])
            constraint2.append({'type': 'edge', 'agent': conflict['a2'], 'loc': loc2, 'time': time})
        elif conflict['type'] == 'agent-block':
            if conflict['block'] == 1:  # a1 is performing block action
                block_agent = conflict['a1']
                move_agent = conflict['a2']
            else:  # a2 is performing block action
                block_agent = conflict['a2']
                move_agent = conflict['a1']
            if conflict['arrive']:
                constraint1.append({'type': 'block', 'agent': block_agent, 'time': time})
            constraint1.append({'type': 'block', 'agent': block_agent, 'time': time + 1})
            constraint2.append({'type': 'vertex', 'agent': move_agent, 'loc': loc, 'time': time})
            if not conflict['arrive']:
                constraint2.append({'type': 'vertex', 'agent': move_agent, 'loc': loc, 'time': time + 1})
        elif conflict['type'] == 'block-block':
            constraint1.append({'type': 'block', 'agent': conflict['a1'], 'time': time})
            constraint2.append({'type': 'block', 'agent': conflict['a2'], 'time': time})
        elif conflict['type'] in ['map', 'height']:
            if conflict['block'] == 1:  # a1 is performing block action
                block_agent = conflict['a1']
                move_agent = conflict['a2']
            else:  # a2 is performing block action
                block_agent = conflict['a2']
                move_agent = conflict['a1']
            if conflict['time'] < conflict['block_time']:  # Block action hasn't happened
                for t in range(conflict['time'], conflict['block_time'] + 1):
                    if conflict['type'] == 'map':
                        constraint1.append({'type': 'edge', 'agent': move_agent, 'loc': loc, 'time': t})
                    else:
                        constraint1.append({'type': 'vertex', 'agent': move_agent, 'loc': loc, 'time': t})
            else:  # Block action already happened
                for t in range(conflict['block_time'], w):
                    if conflict['type'] == 'map':
                        constraint1.append({'type': 'edge', 'agent': move_agent, 'loc': loc, 'time': t})
                    else:
                        constraint1.append({'type': 'vertex', 'agent': move_agent, 'loc': loc, 'time': t})
                for t in range(conflict['time'] + 1):
                    constraint2.append({'type': 'block', 'agent': block_agent, 'time': t})
        return [constraint1, constraint2]

    def construct_height_series(self, infos, ignore_id=-1):
        """
        Construct series of height maps and map changing times according to all agents' block actions
        Notes: changing times are 1 timestep after block actions
        Args:
            infos: list of block action information, (time, location, height change)
            ignore_id: ignore one agent's block action
        """
        temp_info = infos.copy()
        if ignore_id != -1:
            del temp_info[ignore_id]
        temp_info.sort()
        maps = dict()
        times = []
        prev_map = self.height_map.copy()
        prev_t = 0
        maps[prev_t] = prev_map
        times.append(prev_t)
        for info in temp_info:
            if info[-1] != 0:
                t, loc, delta_h = info
                # Map change is offset by 1 timestep
                if t + 1 != prev_t:
                    maps[t + 1] = prev_map.copy()
                    prev_t = t + 1
                    prev_map = maps[t + 1]
                    times.append(t + 1)
                maps[t + 1][loc[0], loc[1]] += delta_h
        return maps, times

    def compute_cost(self, paths, infos):
        """Compute cost of a CT node, based on flow-time (sum of goal completion time)"""
        cost = 0
        for info in infos:
            cost += info[0]
        return cost

    def push_CT_node(self, open_list, node, generate_id):
        """
        Push a CT node to the open list
        Order by: cost (flow time), number of conflicts, note generation id
        """
        heapq.heappush(open_list, (node['cost'], len(node['conflicts']), generate_id, node))

    def append_paths(self, paths, w):
        """Append stay actions to path with length shorter than window size"""
        for i in range(len(paths)):
            if len(paths[i]) < w:
                paths[i] = paths[i].copy() + [(paths[i][-1][0], None)] * (w - len(paths[i]))

    def find_solution(self):
        # TODO: what's the best window size?
        paths, infos = [], []
        shortest = 999
        longest = 0
        maps = {0: self.height_map.copy()}
        for i in range(self.agent_num):
            path, info = self.a_star(i, [], maps, [0], goal_terminate=True, init_run=True)
            if path is None:
                raise Exception('No solution')
            paths.append(path)
            infos.append(info)
            shortest = min(shortest, len(path))
            longest = max(longest, len(path))
        if self.subtask:
            window = longest
        else:
            window = shortest * 2
        self.append_paths(paths, window)

        open_list = []
        generated = expanded = 0
        root = {'paths': paths,
                'infos': infos,
                'constraints': [],
                'cost': self.compute_cost(paths, infos),
                'conflicts': self.detect_all_conflicts(paths, infos, w=window)}
        self.push_CT_node(open_list, root, generate_id=generated)

        self.init_world = self.env.world
        start = time.time()
        while len(open_list) > 0 and time.time() - start < self.timeout:
            node = heapq.heappop(open_list)[-1]
            expanded += 1

            # Check conflicts
            conflicts = node['conflicts']
            if len(conflicts) == 0:
                return node['paths']

            # Resolve one collision
            conflict = conflicts[0]
            window = len(node['paths'][0])
            constraints = self.resolve_conflict(conflict, w=window)
            # Add new child node
            for cons in constraints:
                if cons in node['constraints'] or len(cons) == 0:
                    continue
                child = {'constraints': node['constraints'].copy() + cons.copy(),
                         'paths': node['paths'].copy(), 'infos': node['infos'].copy()}
                aid = cons[0]['agent']
                maps, times = self.construct_height_series(child['infos'], ignore_id=aid)
                result = self.a_star(aid, child['constraints'], maps, times)
                if result is not None:
                    path, info = result
                    prev_len = len(child['paths'][aid])
                    child['paths'][aid] = path
                    child['infos'][aid] = info
                    window = max(len(path), prev_len)
                    self.append_paths(child['paths'], w=window)
                    child['conflicts'] = self.detect_all_conflicts(child['paths'], child['infos'], w=window)
                    child['cost'] = self.compute_cost(child['paths'], child['infos'])
                    if child in open_list:
                        continue
                    generated += 1
                    self.push_CT_node(open_list, child, generate_id=generated)
        print(self.height_map)
        print(self.goal)
        print(self.agent_pos)
        return None

    def plan(self):
        super().plan()
        self.paths = self.find_solution()
        # print('Solution found')

