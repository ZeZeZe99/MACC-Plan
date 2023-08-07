import plotly.graph_objects as go
import numpy as np
import pickle as pk
import config

x0 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
y0 = np.array([0, 1, 1, 0, 0, 1, 1, 0])
z0 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

block_color = {
    0: ['white', 0],         # air
    1: ['cyan', .3],         # unplaced goal block
    2: ['royalblue', 1],     # placed goal block
    3: ['red', .7],          # scaffold block
    4: ['black', 1],         # agent
    5: ['darkgray', .7]      # carried block
}

def cube(x, y, z, block_type=0):
    color, opacity = block_color[block_type]
    return go.Mesh3d(x=x + x0, y=y + y0, z=z + z0, color=color, opacity=opacity, alphahull=0)

def init_world():
    world = [cube(x, y, z, block_type=0)
             for z in range(h) for x in range(w) for y in range(w)]
    return world

def coord2id(z, x, y):
    return z * w * w + x * w + y

def init_frame():
    world = init_world()
    traces = np.arange(w * w * h).tolist()
    return go.Frame(data=world, traces=traces, name=f'frame {-1}')

def update_frame(frame_id, new_map, prev_frame, prev_map):
    diff = new_map != prev_map
    z, x, y = np.nonzero(diff)
    traces = list(prev_frame.traces).copy()
    world = list(prev_frame.data).copy()
    for i in range(len(x)):
        idx = coord2id(z[i], x[i], y[i])
        world[idx] = cube(x[i], y[i], z[i], new_map[z[i], x[i], y[i]])
    return go.Frame(data=world, traces=traces, name=f'frame {frame_id}')

def convert3d(ob):
    map3d = np.zeros((h, w, w))
    for i in range(h):
        block = ob[0] > i
        goal = ob[1] > i
        map3d[i] = np.where(block > goal, 3, 0)
        map3d[i] = np.where(block < goal, 2, map3d[i])
        map3d[i] = np.where((block == goal) & (block > 0), 1, map3d[i])
    return map3d

def update_layout(fig, length):
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        # eye=dict(x=1, y=2.2, z=1)  # Plan only
        eye=dict(x=1, y=0.2, z=2.3)
    )
    updatemenus = [dict(
        buttons=[
            dict(
                args=[None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True}],
                label="Play",
                method="animate"
            ),
            dict(
                args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                               "transition": {"duration": 0}}],
                label="Pause",
                method="animate"
            )
        ],
        direction="left", pad={"r": 10, "t": 87},
        showactive=False, type="buttons",
        x=0.1, xanchor="right", y=0.3, yanchor="top"
    )]
    sliders = [dict(
        steps=[
            dict(method='animate',
                 args=[[f'frame {k}'],
                       dict(mode='immediate',
                            frame=dict(duration=400, redraw=True),
                            transition=dict(duration=0)
                            )
                       ],
                 label=f'{k}'
                 ) for k in range(length + 1)
        ],
        active=0,
        transition=dict(duration=0),
        x=0,  # slider starting position
        y=0,
        currentvalue=dict(font=dict(size=12),
                          prefix='frame: ',
                          visible=True,
                          xanchor='center'
                          ),
        len=1.0  # slider length
    )]
    scene = dict(zaxis=dict(showticklabels=False))
    fig.update_layout(updatemenus=updatemenus, sliders=sliders, scene_camera=camera, scene=scene)

def create_gif():
    fig = go.Figure(data=init_world())
    frames = [init_frame()]
    prev_map = np.zeros((h, w, w))
    for i in range(len(maps)):
        new_map = maps[i]
        frames.append(update_frame(i + 1, new_map, frames[-1], prev_map))
        prev_map = new_map
    fig.update(frames=frames)
    update_layout(fig, len(maps))
    fig.show()

def convert_high_plan(goal, plan):
    """Convert a high level plan to a sequence of frames"""
    goal3d = np.zeros((h, w, w), dtype=np.int8)
    for i in range(h):
        goal3d[i] = goal > i
    frames = np.tile(goal3d, (len(plan) + 1, 1, 1, 1))
    for i in range(len(plan)):
        add, x, y, lv = plan[i]
        scaffold = 1 - goal3d[lv, x, y]
        if add:
            frames[i+1:, lv, x, y] += 1 + 2 * scaffold
        else:
            frames[i+1:, lv, x, y] -= 1 + 2 * scaffold
    return frames

def convert_path(goal, plan):
    """Convert paths of multiple agents to a sequence of frames"""
    goal3d = np.zeros((h, w, w), dtype=np.int8)
    height = np.zeros((w, w), dtype=np.int8)
    num = len(plan)
    time = len(plan[0])
    for i in range(h):
        goal3d[i] = goal > i
    frames = np.tile(goal3d, (time + 1, 1, 1, 1))
    for t in range(time):
        for i in range(num):
            loc, carry, g = plan[i][t]
            if len(loc) == 2:
                x, y = loc
                lv = height[x, y]
            else:
                x, y, lv = loc
            if x != -1:
                frames[t+1, lv, x, y] = 4
                if carry:
                    frames[t+1, lv+1, x, y] = 5
            if g is not None:
                add, gx, gy, glv = g
                scaffold = 1 - goal3d[glv, gx, gy]
                if add:
                    frames[t+1:, glv, gx, gy] += 1 + 2 * scaffold
                    height[gx, gy] += 1
                else:
                    frames[t+1:, glv, gx, gy] -= 1 + 2 * scaffold
                    height[gx, gy] -= 1
    return frames


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()
    high = arg.high

    if high:
        load_path = f'result/high_action_{arg.map}.pkl' if arg.map > 0 else 'result/high_action.pkl'
        with open(load_path, 'rb') as f:
            plan, info = pk.load(f)
            goal = info['goal']
            h = np.max(goal)
    else:
        load_path = f'result/path_{arg.map}.pkl' if arg.map > 0 else 'result/path.pkl'
        with open(load_path, 'rb') as f:
            goal, plan = pk.load(f)
            h = np.max(goal) + 2

    w = goal.shape[0]
    if high:
        maps = convert_high_plan(goal, plan)
    else:
        maps = convert_path(goal, plan)
    create_gif()
