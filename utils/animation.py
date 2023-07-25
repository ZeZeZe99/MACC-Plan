import plotly.graph_objects as go
import numpy as np
import pickle as pk

x0 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
y0 = np.array([0, 1, 1, 0, 0, 1, 1, 0])
z0 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

block_color = {
    0: ['white', 0],           # air
    1: ['yellow', 1],         # correct block
    2: ['springgreen', .3],   # plan
    3: ['red', .7],           # scaffolding
}

def cube(x, y, z, block_type=0):
    color, opacity = block_color[block_type]
    return go.Mesh3d(x=x + x0, y=y + y0, z=z + z0, color=color, opacity=opacity, alphahull=0)

def init_world():
    world = [cube(x, y, z, block_type=0)
             for z in range(h) for x in range(w) for y in range(w)]
    return world

def create_heatmap(data=None):
    if data is None:
        data = np.zeros((w, w))
    else:
        data = data.reshape((w, w)).round(decimals=2)
    x = np.arange(w)
    y = np.arange(w)
    text = data.astype(str)
    return go.Heatmap(x=x, y=y, z=data, text=text, colorscale='Blues', zmin=0, zmax=1, texttemplate='%{text}')

def coord2id(z, x, y):
    return z * w * w + x * w + y

def init_frame():
    world = init_world()
    heatmap = create_heatmap()
    traces = np.arange(w * w * h + 2).tolist()
    return go.Frame(data=world + [heatmap, heatmap], traces=traces, name=f'frame {-1}')

def update_frame(frame_id, new_map, new_pi, prev_frame, prev_map):
    diff = new_map != prev_map
    z, x, y = np.nonzero(diff)
    traces = list(prev_frame.traces).copy()
    world = list(prev_frame.data).copy()
    for i in range(len(x)):
        idx = coord2id(z[i], x[i], y[i])
        world[idx] = cube(x[i], y[i], z[i], new_map[z[i], x[i], y[i]])
    if new_pi is None:
        world[-2] = create_heatmap()
        world[-1] = create_heatmap()
    else:
        new_pi = new_pi.reshape(2, w, w)
        world[-2] = create_heatmap(new_pi[0])
        world[-1] = create_heatmap(new_pi[1])
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
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['yaxis5']['autorange'] = "reversed"
    fig['layout']['xaxis']['side'] = "top"
    fig['layout']['xaxis5']['side'] = "top"

def create_gif(episode):
    maps, policies = episode
    fig = go.Figure(data=init_world())
    fig.set_subplots(rows=3, cols=4)
    fig.add_trace(create_heatmap(), row=1, col=1)
    fig.add_trace(create_heatmap(), row=2, col=1)
    frames = [init_frame()]
    prev_map = np.zeros((h, w, w))
    for i in range(len(maps)):
        new_map = convert3d(maps[i])
        new_pi = policies[i]
        frames.append(update_frame(i + 1, new_map, new_pi, frames[-1], prev_map))
        prev_map = new_map
    fig.update(frames=frames)
    update_layout(fig, len(maps))
    fig.show()

if __name__ == '__main__':
    w = 8
    h = 3

    with open('frames.pkl', 'rb') as f:
        episodes = pk.load(f)
    for episode in episodes:
        create_gif(episode)
