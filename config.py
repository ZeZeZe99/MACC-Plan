import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Planning settings')

    '''Environment'''
    parser.add_argument('--h', type=int, default=5, help='Height of the grid world')
    parser.add_argument('--w', type=int, default=8, help='Width of the grid world')
    parser.add_argument('--map', type=int, default=0, help='Goal map')
    parser.add_argument('--num', type=int, default=1, help='Number of agents')
    parser.add_argument('--cost', type=float, default=.1, help='Base cost of an action')

    '''Network'''
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension')

    '''Learning'''
    parser.add_argument('--algo', type=str, default='A2C', help='RL algorithm')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--clip', type=float, default=10, help='Gradient clipping')
    parser.add_argument('--entropy', type=float, default=0.2, help='Entropy loss coefficient')

    parser.add_argument('--episode', type=int, default=10000, help='Episode number')
    parser.add_argument('--epi_len', type=int, default=100, help='Episode length')
    parser.add_argument('--train_step', type=int, default=100, help='Training frequency (per _ steps)')
    parser.add_argument('--test_freq', type=int, default=1000, help='Testing frequency (per _ episodes)')
    parser.add_argument('--test_epi', type=int, default=10, help='Testing episode number')
    parser.add_argument('--save_freq', type=int, default=10000, help='Saving frequency (per _ episodes)')
    parser.add_argument('--save_dir', type=str, default='', help='Directory for model saving')

    '''Logging'''
    parser.add_argument('--wandb', action='store_true', help='Log with wandb')
    parser.add_argument('--comment', type=str, default='', help='Comment for the run')

    return parser

def process(arg):
    arg.num_action = arg.w * arg.w * 2

    arg.project = 'simple'
    arg.group = f'w{arg.w}'
    arg.name = f'{arg.algo}-{arg.comment}'

    if arg.save_dir == '':
        arg.save_dir = f'save/{arg.project}/{arg.group}/'

    return arg