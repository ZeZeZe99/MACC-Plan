import argparse

m = 6

teleport = True
heu = 2
order = 5

cost = 1
detect = 0
resolve = 3

reselect = False
allocate = 1
k = 1
execute = 2

def get_parser():
    parser = argparse.ArgumentParser(description='Planning settings')

    '''Environment'''
    parser.add_argument('--h', type=int, default=5, help='Height of the grid world')
    parser.add_argument('--w', type=int, default=8, help='Width of the grid world')
    parser.add_argument('--map', type=int, default=m, help='Goal map')
    parser.add_argument('--num', type=int, default=1, help='Number of agents')

    '''Animation'''
    parser.add_argument('--gif', action='store_true', help='Save gif of the episode')
    parser.add_argument('--high', action='store_true', help='High level animation')

    '''Hierarchical planning'''
    parser.add_argument('--start', type=int, default=0, help='Start stage in the pipeline')

    '''Planning parameters'''
    parser.add_argument('--symmetry', type=int, default=0, help='High level symmetry detection mode')
    parser.add_argument('--high_order', type=int, default=1, help='High level A* node order mode')
    parser.add_argument('--high_heu', type=int, default=1, help='High level heuristic mode')
    parser.add_argument('--valid', type=int, default=1, help='High level valid degree')

    parser.add_argument('--teleport', type=bool, default=teleport, help='Teleport mode')
    parser.add_argument('--heu', type=int, default=heu, help='Low level heuristic mode')
    parser.add_argument('--order', type=int, default=order, help='Low level A* node order mode')

    parser.add_argument('--cost', type=int, default=cost, help='CBS cost mode')
    parser.add_argument('--detect', type=int, default=detect, help='CBS conflict detection order')
    parser.add_argument('--resolve', type=int, default=resolve, help='CBS conflict resolution order')

    parser.add_argument('--select', type=int, default=1, help='Task selection mode')
    parser.add_argument('--reselect', type=bool, default=reselect, help='Re-select tasks upon each round of allocation')
    parser.add_argument('--allocate', type=int, default=allocate, help='Task allocation mode')
    parser.add_argument('--k', type=int, default=k, help='Levels of tasks to allocate')
    parser.add_argument('--execute', type=int, default=execute, help='Re-plan upon execution of first task')

    '''Profile'''
    parser.add_argument('--profile', action='store_true', help='Profile runtime')

    return parser

def process_config():
    arg = get_parser()
    arg = arg.parse_args()

    # if arg.resolve == 3:
    #     arg.priority = {'edge-block': 0, 'agent-block': 0, 'move-height': 0, 'block-height': 0, 'vertex': 1, 'edge': 1}
    # elif arg.resolve == 4:
    #     arg.priority = {'edge-block': 0, 'agent-block': 0, 'move-height': 1, 'block-height': 1, 'vertex': 2, 'edge': 2}
    # elif arg.resolve == 5:
    #     arg.priority = {'edge-block': 1, 'agent-block': 1, 'move-height': 0, 'block-height': 0, 'vertex': 2, 'edge': 2}
    # else:
    #     arg.priority = {'edge-block': 0, 'agent-block': 0, 'move-height': 0, 'block-height': 0, 'vertex': 0, 'edge': 0}
    if arg.resolve == 3:
        arg.priority = {'prior-height': 0, 'edge-block': 0, 'agent-block': 0, 'height': 0, 'vertex': 1, 'edge': 1}
    elif arg.resolve == 4:
        arg.priority = {'prior-height': 0, 'edge-block': 1, 'agent-block': 1, 'height': 1, 'vertex': 2, 'edge': 2}
    else:
        arg.priority = {'prior-height': 0, 'edge-block': 0, 'agent-block': 0, 'height': 0, 'vertex': 0, 'edge': 0}
    return arg
