import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Planning settings')

    '''Environment'''
    parser.add_argument('--h', type=int, default=5, help='Height of the grid world')
    parser.add_argument('--w', type=int, default=8, help='Width of the grid world')
    parser.add_argument('--map', type=int, default=0, help='Goal map')
    parser.add_argument('--num', type=int, default=1, help='Number of agents')

    '''Animation'''
    parser.add_argument('--gif', action='store_true', help='Save gif of the episode')

    '''Hierarchical planning'''
    parser.add_argument('--start', type=int, default=0, help='Start stage in the pipeline')

    return parser
