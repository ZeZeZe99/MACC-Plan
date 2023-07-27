import gurobi as grb
import numpy as np
import ast
import pickle as pk

import lego
import config

"""
A flow formulation to find a high-level plan to construct the goal structure
Solved via Linear Programming

Variables:
    1. 2d edges: f(u, v, t) = flow from u to v at time t = move action
    2. 2d vertices: h(u, t) = height of u at time t
    3. 3d edges: f(a, u, t, v) = action from u to v at time t, edge is connected from (u, t) to (u, t')
        3.1 Optional noop action
    All border locations are merged into a single border vertex B = (-1, -1)
Constraints:
    1. Flow conservation: in-flow = out-flow
    2. Source flow: f(S, (B, 0)) = 1
    3. Sink flow: f((B, T), T) = 1
    4. Return to border: flow through B has to be 1 for each time step
    5. Move action: if f(u, v, t) = 1, then |h(u, t) - h(v, t)| <= 1 (height difference is at most 1)
    6. Block action: Σ_u,v a(u, t, v) = 1 (only one block action at each time step)
    7. Add action: if f(add, u, t, v) = 1, then h(u, t) = h(v, t) (equal height)
    8. Remove action: if f(remove, u, t, v) = 1, then h(v, t) = h(u, t) + 1 (target height is one higher)
    9. Height change: h(v, t + 1) = h(v, t) + Σ_u f(add, u, t, v) - Σ_u f(remove, u, t, v)
    10. Shadow region: h(u, t) <= shadow height at u
Objective:
    1. 0, find a feasible solution
    2. Maximize # noops
"""


def flow(env, T, relax=False, noop=False):
    max_h = np.max(env.goal)
    workspace = (env.shadow_height > 0) * 1

    # 2D edges and vertices
    vertex = env.valid_loc - env.border_loc
    vertex.add((-1, -1))  # Represent all border locations
    edge = set()
    for u in env.valid_loc:
        if u in env.border_loc:
            continue
        for v in env.search_neighbor[u]:
            edge.add((u, v))
        if u in env.start_loc:
            edge.add(((-1, -1), u))
            edge.add((u, (-1, -1)))

    # Dictionary for variables
    in_edges = dict()
    out_edges = dict()
    action_edges = dict()
    edge_by_time = dict()
    action_by_time = dict()
    adds = dict()
    removes = dict()
    noops = dict()
    heights = dict()
    for t in range(T + 1):
        for v in vertex:
            in_edges[t, v] = []
            out_edges[t, v] = []
            action_by_time[t] = []
            if workspace[v] == 1:
                adds[t, v] = []
                removes[t, v] = []

    '''Create model'''
    model = grb.Model()
    if relax:
        model.setParam('OutputFlag', 0)

    '''Create variables'''
    for t in range(T + 1):
        '''Height for each non-border location'''
        for v in vertex:
            if workspace[v] == 0:
                continue
            heights[t, v] = model.addVar(lb=0, ub=max_h, vtype=grb.GRB.INTEGER, name=f'h_{v, t}', obj=0, column=None)
        '''Edge between neighboring locations = move action'''
        for u, v in edge:
            e = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'e_{u, v, t}', obj=0, column=None)
            in_edges[t, v].append(e)
            out_edges[t, u].append(e)
            edge_by_time[t, u, v] = e

    for t in range(T):
        '''Action variables (add/remove) from u to v, u = agent location, v = target location'''
        for u, v in edge:
            if workspace[v] == 0:  # Skip add or remove to non-shadow location
                continue
            e_add = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'{1, u, v, t}', obj=0, column=None)
            e_remove = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'{0, u, v, t}', obj=0, column=None)
            action_by_time[t].extend([e_add, e_remove])
            out_edges[t, u].extend([e_add, e_remove])  # Agent location, t -> t + 1
            in_edges[t + 1, u].extend([e_add, e_remove])
            adds[t, v].append(e_add)
            removes[t, v].append(e_remove)
            action_edges[t, u, v] = [e_add, e_remove]
        '''Dummy action variables: no-op at border locations'''
        if noop:
            e_noop = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'e_noop_{t}', obj=0, column=None)
            out_edges[t, (-1, -1)].append(e_noop)
            in_edges[t + 1, (-1, -1)].append(e_noop)
            action_by_time[t].append(e_noop)
            noops[t] = e_noop

    '''Edge from source to (0, border) and (T, border) to sink'''
    e_source = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'e_source', obj=0, column=None)
    in_edges[0, (-1, -1)].append(e_source)
    e_sink = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'e_sink', obj=0, column=None)
    out_edges[T, (-1, -1)].append(e_sink)

    '''Create constraints'''
    '''Source and sink'''
    model.addConstr(e_source == 1, name='source')
    model.addConstr(e_sink == 1, name='sink')

    '''Initial height'''
    model.addConstrs((heights[0, v] == 0 for v in vertex if workspace[v] == 1), name='init_height')

    '''Flow conservation: in-flow = out-flow'''
    model.addConstrs((grb.quicksum(in_edges[t, v]) == grb.quicksum(out_edges[t, v])
                      for t in range(T + 1) for v in vertex), name='flow_conservation')

    '''Return to border: make sure plan will be feasible'''
    model.addConstrs((grb.quicksum(in_edges[(t, (-1, -1))]) == 1 for t in range(T + 1)), name='border')

    '''Move: height difference <= 1'''
    for t, u, v in edge_by_time:
        if workspace[u] == 0 and workspace[v] == 0:
            continue
        hu = 0 if workspace[u] == 0 else heights[t, u]
        hv = 0 if workspace[v] == 0 else heights[t, v]
        model.addGenConstrIndicator(edge_by_time[t, u, v], True, hu - hv <= 1, name='valid_move')
        model.addGenConstrIndicator(edge_by_time[t, u, v], True, hv - hu <= 1, name='valid_move')

    '''Block action: only one action per time'''
    model.addConstrs((grb.quicksum(action_by_time[t]) == 1 for t in range(T)), name='action')

    '''Block action: relative height should be correct'''
    for t, u, v in action_edges:
        hu = 0 if workspace[u] == 0 else heights[t, u]
        # Add: h(u) should be equal to h(v); add(u, v) = 1 => h(u) = h(v)
        model.addGenConstrIndicator(action_edges[t, u, v][0], True, hu == heights[t, v], name='valid_add')
        # Remove: h(v) should be equal to h(u) + 1; remove(u, v) = 1 => h(v) = h(u) + 1
        model.addGenConstrIndicator(action_edges[t, u, v][1], True, heights[t, v] == hu + 1, name='valid_remove')

    '''Height change'''
    model.addConstrs((heights[t + 1, v] == heights[t, v] + grb.quicksum(adds[t, v]) - grb.quicksum(removes[t, v])
                      for t in range(T) for v in vertex if workspace[v] == 1), name='height_change')

    '''Goal completion: final height = goal height'''
    model.addConstrs((heights[T, v] == env.goal[v] for v in vertex if workspace[v] == 1), name='goal_completion')

    '''Shadow region'''
    model.addConstrs((heights[t, v] <= env.shadow_height[v] for t in range(T + 1) for v in vertex if workspace[v] == 1),
                     name='shadow')

    '''Set objective'''
    if noop:
        model.setObjective(grb.quicksum(noops[t] for t in range(T)), grb.GRB.MAXIMIZE)
    else:
        model.setObjective(0)

    '''Relax: check feasibility'''
    if relax:
        relaxed_model = model.copy()
        relaxed_model.feasRelaxS(1, False, False, False)
        relaxed_model.optimize()
        return relaxed_model.status != grb.GRB.INFEASIBLE

    '''Solve'''
    model.optimize()

    '''Solution'''
    final_height = np.zeros(env.world_shape, dtype=np.int8)
    for v in vertex:
        if workspace[v] == 0:
            continue
        final_height[v] = heights[T, v].x
    print(final_height)
    action_sequence = []
    for t in range(T):
        if noop and noops[t].x > 0:
            continue
        for a in action_by_time[t]:
            if a.x > 0:
                add, nbr, loc, _ = ast.literal_eval(a.varName)
                z = int(heights[t, nbr].x) if workspace[nbr] == 1 else 0
                action_sequence.append((add, loc[0], loc[1], z))
                break
    print(action_sequence)
    return action_sequence


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.set_goal()
    env.set_shadow()
    high_actions = flow(env, 33, relax=False, noop=False)

    with open('result/high_action.pkl', 'wb') as f:
        pk.dump([env.goal, high_actions, {'valid': [], 'shadow': env.shadow}], f)
