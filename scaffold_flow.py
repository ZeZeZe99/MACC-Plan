import gurobi as grb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import lego
import config

"""
A min-cost flow formulation to find valid paths to construct each tower
Solved via Linear Programming

Variables:
    1. 3d edges: f(u, z, v, z') = flow from u at height z to v at height z'
    2. 3d blocks: b(u, z) = usage of block (u, z)
    3. 2d edges: e(u, v) = usage of any 3d edge from u to v
    4. 2d vertices: o(u) = order of vertex u
    5. 2d vertices: c(u) = cost to add extra scaffold at u
Constraints:
    1. Flow conservation: in-flow = out-flow for each vertex
    2. Source flow: 1 unit of in-flow to the top vertex (block) of each tower
    3. Block usage: b(u, z) = 1 iff flow through (u, z) > 0
    4. Edge usage: e(u, v) = 1 iff f(u, z, v, z') > 0 for some z, z'
    5. Edge conflict: e(u, v) + e(v, u) <= 1
    6. Cycle avoidance / order: o(u) > o(v) if e(u, v) = 1
    7. Cost: c(u) = max_z c(u, z) * b(u, z) = maximum # scaffold required at u
    Optional:
    8. Goal removal: for g(u) < g(v), if e(u, v) = 1 then c(u) <= c(v) + 1 (avoid removing goal blocks)
    9. Out-degree: out-degree(u) <= 1 (enforce tree-like paths, not included yet)
Objective:
    Minimize # scaffold used = Σ_u c(u)
"""


def add_v_dummy(model, env, edge3d, edge2d, in_out_edge, n):
    """Generate a dummy level of 3d vertices (z = -1). Border vertices are considered as sinks"""
    work_loc = env.valid_loc - env.border_loc
    for x, y in work_loc:
        in_out_edge[(x, y, -1)] = [[], []]
    for x, y in work_loc:
        for x2, y2 in env.search_neighbor[(x, y)]:
            e = model.addVar(lb=0, ub=n, name=f'f{x,y,-1,x2,y2,-1}', vtype=grb.GRB.INTEGER, obj=0, column=None)
            edge3d[((x, y, -1), (x2, y2, -1))] = e
            edge2d[((x, y), (x2, y2))] = [e]
            in_out_edge[(x, y, -1)][1].append(e)
            in_out_edge[(x2, y2, -1)][0].append(e)
        # For vertices next to border: add dummy edge to sink
        if (x, y) in env.start_loc:
            e = model.addVar(lb=0, ub=n, name=f'f{x,y,-1},sink', vtype=grb.GRB.INTEGER, obj=0, column=None)
            in_out_edge[(x, y, -1)][1].append(e)

def add_v_3d(model, edge3d, edge2d, in_out_edge, n, x, y, z, visited):
    """Recursively generate edge variables, starting from (x, y, z) and add its neighbors at z-1"""
    edges = []
    if (x, y, z) in visited:
        return visited[(x, y, z)]
    if z == 0:  # Ground level, connect to dummy level right below
        e = model.addVar(lb=0, ub=n, name=f'f{x,y,0,x,y,-1}', vtype=grb.GRB.INTEGER)
        edges.append(e)
        valid = True
        in_out_edge[(x, y, -1)][0].append(e)
        edge3d[((x, y, 0), (x, y, -1))] = e
    else:  # Higher levels, connect to 1 level lower
        valid = False
        for (x2, y2) in env.search_neighbor[(x, y)]:
            if add_v_3d(model, edge3d, edge2d, in_out_edge, n, x2, y2, z - 1, visited):
                e = model.addVar(lb=0, ub=n, name=f'f{x,y,z,x2,y2,z-1}', vtype=grb.GRB.INTEGER)
                edges.append(e)
                valid = True
                '''Record in edge of vertex (x2, y2, z-1)'''
                if (x2, y2, z - 1) not in in_out_edge:
                    in_out_edge[(x2, y2, z - 1)] = [[], []]
                in_out_edge[(x2, y2, z - 1)][0].append(e)
                edge3d[(x, y, z), (x2, y2, z - 1)] = e
                '''Record edge that uses 2d vertices (x,y), (x2, y2)'''
                u, v = (x, y), (x2, y2)
                edge2d[(u, v)].append(e)
    visited[(x, y, z)] = valid
    if valid:
        '''Record out edge of vertex (x, y, z)'''
        if (x, y, z) not in in_out_edge:
            in_out_edge[(x, y, z)] = [[], []]
        in_out_edge[(x, y, z)][1].extend(edges)
    return valid

def min_cost_flow(env, height, removal=False):
    """Solve the minimum cost flow problem to find the optimal scaffold"""
    workspace = np.clip(height, 0, 1)
    max_h = np.max(height)
    num = workspace.sum()

    '''Cost matrix: cost[z, x, y] = scaffold required to reach level z at (x, y)'''
    cost = np.zeros((max_h, *env.world_shape), dtype=np.int8)
    for z in range(max_h):
        cost[z] = np.clip(z + 1 - height, 0, max_h)

    edges2d = dict()  # Map from 2d edge to all 3d edges that use it
    in_out_edges = dict()  # Map from 3d vertex to its in/out edges
    block_var = dict()
    edge_var = [dict(), dict()]  # 3d edge, 2d edge
    vertex_var = dict()
    cost_var = dict()

    model = grb.Model()

    '''3D edge variables: f(u, v) = flow from u to v, where u, v are 3d vertices'''
    # Dummy level (z = -1)
    add_v_dummy(model, env, edge_var[0], edges2d, in_out_edges, num)
    # Other levels (z >= 0)
    visited = dict()
    for (x, y) in np.transpose(np.nonzero(workspace)):
        z = height[x, y] - 1
        valid = add_v_3d(model, edge_var[0], edges2d, in_out_edges, num, x, y, z, visited)
        assert valid

    '''Flow conservation constraints: in-flow = out-flow'''
    source_edges = []
    for loc3d in in_out_edges:
        # Connect top of each goal tower to a dummy source
        if loc3d[2] != -1 and height[loc3d[0], loc3d[1]] == loc3d[2] + 1:
            source_e = model.addVar(lb=0, ub=1, name=f'source_{loc3d}', vtype=grb.GRB.BINARY, obj=0, column=None)
            in_out_edges[loc3d][0].append(source_e)
            source_edges.append(source_e)
        model.addConstr(grb.quicksum(in_out_edges[loc3d][0]) == grb.quicksum(in_out_edges[loc3d][1]), name=f'flow_{loc3d}')

    '''Source constraint: flow = 1 for each source (group)'''
    model.addConstrs((source_edges[i] == 1 for i in range(len(source_edges))), name='source_{i}')

    '''Block variables: b(x, y, z) = usage of block at (x, y, z)
       b(x, y, z) = 1 iff flow through (x, y, z) is positive: Σ_u f(v, u) >= 1, v = (x, y, z)'''
    M = 10 * num
    for (x, y, z) in in_out_edges:
        if z == -1:  # Skip dummy level
            continue
        b = model.addVar(lb=0, ub=1, name=f'block_{x,y,z}', vtype=grb.GRB.BINARY, obj=0, column=None)
        # b = 1 <=> Σ_u f(v, u) >= 1
        model.addConstr(grb.quicksum(in_out_edges[(x, y, z)][1]) <= M * b, name=f'block_{x,y,z}')
        model.addConstr(grb.quicksum(in_out_edges[(x, y, z)][1]) >= M * (b-1) + 1, name=f'block_{x,y,z}')
        if (x, y) not in block_var:
            block_var[(x, y)] = dict()
        block_var[(x, y)][z] = b

    '''2D edge variables: e(u, v) = usage of any lv-edge from u to v, where u, v are 2d vertices
       e(u, v) = 1 iff Σ_f(u, v) >= 1, f(u, v) is 3d edge that uses (u, v)'''
    for u, v in edges2d:
        e = model.addVar(lb=0, ub=1, name=f'e_{u}_{v}', vtype=grb.GRB.BINARY, obj=0, column=None)
        edge_var[1][(u, v)] = e
        # e = 1 <=> Σ_f(u, v) >= 1
        model.addConstr(grb.quicksum(edges2d[(u, v)]) <= M * e, name=f'edge_{u, v}')
        model.addConstr(grb.quicksum(edges2d[(u, v)]) >= M * (e-1) + 1, name=f'edge_{u, v}')

    '''2D edge constraint: no edge conflict'''
    done = set()
    for u, v in edges2d:
        if (v, u) not in edges2d:
            continue
        if (v, u) in done:
            continue
        done.add((u, v))
        model.addConstr(edge_var[1][(u, v)] + edge_var[1][(v, u)] <= 1, name=f'edge_{u, v}')

    '''2D cycle constraint: no cycle in the 2d graph
       Each 2d vertex is given an order. o(u) >= o(v) + 1 if e(u, v) = 1'''
    M = env.w * env.w
    for v in env.valid_loc:
        vertex_var[v] = model.addVar(lb=0, ub=grb.GRB.INFINITY, name=f'o_{v}', vtype=grb.GRB.INTEGER, obj=0, column=None)
    for u, v in edges2d:
        model.addConstr(vertex_var[u] + M * (1 - edge_var[1][(u, v)]) >= vertex_var[v] + 1, name=f'order_{u, v}')

    '''Cost variables: c(x, y) = cost for adding scaffolds at (x, y)
       Cost constraint: c(x, y) = max_z (c(x, y, z) * b(x, y, z, g))'''
    for loc in block_var:
        c = model.addVar(lb=0, ub=max_h, name=f'cost_{loc}', vtype=grb.GRB.INTEGER, obj=0, column=None)
        model.addConstrs((c >= cost[z][loc] * block_var[loc][z] for z in block_var[loc]), name=f'cost_{loc}')
        cost_var[loc] = c

    '''Goal removal constraint: disallow removing goal blocks
       If e(u, v) = 1 then c(u) <= c(v) + 1 to avoid removing goal blocks at v'''
    if removal:
        for u, v in edges2d:
            if env.goal[u] < env.goal[v] and u in block_var and v in block_var:
                model.addConstr(cost_var[u] <= M * (1 - edge_var[1][(u, v)]) + 1, name=f'remove_goal_{u, v}')

    '''Goal removal penalty'''
    # for u, v in edges2d:
    #     if env.goal[u] < env.goal[v] and u in block_var and v in block_var:
    #         used_g = model.addVar(lb=0, ub=max_h, name=f'used_goal_{v}', vtype=grb.GRB.INTEGER, obj=0, column=None)
    #         model.addGenConstrMin(used_g, [cost_var[u], env.goal[v] + 1], name=f'used_goal_{v}', constant=0)
    #         pen = model.addVar(lb=0, ub=max_h, name=f'penalty_{u}_{v}', vtype=grb.GRB.INTEGER, obj=0, column=None)
    #         model.addGenConstrMax(pen, [used_g - 1 - env.goal[u], 0], name=f'penalty_{u}_{v}', constant=0)

    '''Objective: minimize cost'''
    model.setObjective(grb.quicksum(cost_var.values()), grb.GRB.MINIMIZE)

    '''Solve'''
    model.optimize()
    # model.printStats()

    '''Solution: scaffold to add at each location'''
    scaffold = np.zeros(env.world_shape, dtype=np.int8)
    for loc in cost_var:
        if cost_var[loc].X > 0:
            val = round(cost_var[loc].X)
            scaffold[loc] = val
    '''Solution: 2d & 3d edges used'''
    edges = [set(), set()]
    for u, v in edge_var[0]:
        if edge_var[0][(u, v)].X > 0:
            edges[0].add((u, v))
    for u, v in edge_var[1]:
        if edge_var[1][(u, v)].X > 0:
            edges[1].add((u, v))

    return scaffold, edges

def propose_scaffold(removal=False):
    height = env.goal.copy()
    scaffold, edges = min_cost_flow(env, height, removal=removal)
    height += scaffold
    print(height)
    print(f'Scaffold: {scaffold.sum()}')
    return height, edges

def create_flow_graph(edges):
    flow_3d, flow_2d = nx.DiGraph(), nx.DiGraph()
    for u, v in edges[0]:
        if u[2] == -1 or v[2] == -1:
            continue
        # Add 3d edges
        flow_3d.add_edge(u, v)
        # Add 2d edges
        flow_2d.add_edge(u[:2], v[:2])
    # Add source and sink nodes to 3d flow graph
    sources_3d, sinks_3d = [], []
    for n in flow_3d.nodes:
        if flow_3d.in_degree[n] == 0:
            sources_3d.append(n)
        if flow_3d.out_degree[n] == 0:
            sinks_3d.append(n)
    flow_3d.add_edges_from([('S', n) for n in sources_3d])
    flow_3d.add_edges_from([(n, 'T') for n in sinks_3d])
    # Add source and sink nodes to 2d flow graph
    sources_2d, sinks_2d = [], []
    for n in flow_2d.nodes:
        if flow_2d.in_degree[n] == 0:
            sources_2d.append(n)
        if flow_2d.out_degree[n] == 0:
            sinks_2d.append(n)
    flow_2d.add_edges_from([('S', n) for n in sources_2d])
    flow_2d.add_edges_from([(n, 'T') for n in sinks_2d])
    return flow_3d, flow_2d

def create_build_graph(edges, height):
    bg = nx.DiGraph()
    for u, v in edges[0]:
        # Add nodes and edges to block graph
        (x, y, z), (x2, y2, z2) = u, v
        # Vertical edges to block right below
        bg.add_edges_from([((x, y, j + 1), (x, y, j)) for j in range(z)])
        bg.add_edges_from([((x2, y2, j + 1), (x2, y2, j)) for j in range(z2)])
        # Diagonal edges to neighbor below
        bg.add_edges_from([((x, y, j + 1), (x2, y2, j)) for j in range(z)])
        # Horizontal edges from v to u
        bg.add_edges_from([((x2, y2, j), (x, y, j)) for j in range(1, z)])
    # Extra edge for block graph
    for u, v in edges[1]:
        if height[u] == height[v]:
            x, y, x2, y2, z = u[0], u[1], v[0], v[1], height[u] - 1
            bg.add_edge((x2, y2, z), (x, y, z))
    draw_graph(bg)
    return bg

def draw_graph(graph):
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='skyblue', edge_color='black',
            width=1.5, alpha=0.7)
    plt.show()


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    env = lego.GridWorld(arg)
    env.set_goal()

    height, edges = propose_scaffold(removal=False)
    fg3d, fg2d = create_flow_graph(edges)
    # draw_graph(fg3d)
    # draw_graph(fg2d)
