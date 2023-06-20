import os.path
import cProfile
import torch
import wandb
import pickle as pk

import config
import lego
import agent

def play(train=True):
    """Play for an episode"""
    '''Initialize variables'''
    step = 0
    done = False
    epi_r = 0
    frames = []
    policies = []

    '''Reset environment and agent'''
    s = env.reset()
    agent.reset()
    if arg.gif:
        frames.append(s)
        policies.append(None)

    '''Start the episode'''
    while not done and step < arg.epi_len:
        valid_map = env.valid_bfs_map(s[0], degree=1)
        valid = valid_map[1:].reshape(-1)
        action, policy = agent.select_action(s, valid)
        s2, r, done = env.step(action)
        if train:
            agent.store_transition(s, action, r, s2, int(done), valid)
        epi_r += r
        s = s2
        step += 1

        # Update network
        if train and (step % arg.train_step == 0 or done or step == arg.epi_len):
            agent.update()
            agent.buffer.reset()

        if arg.gif:
            frames.append(s2)
            policies.append(policy.numpy())

    '''End of episode summary'''
    status = env.status(env.height)
    status['reward'] = epi_r
    status['done'] = int(done)
    status['makespan'] = step
    return status, frames, policies

def train(count):
    stat = play()[0]
    print(f'Train episode {count}: done {stat["done"]}, makespan {stat["makespan"]}, reward {stat["reward"]},'
          f' completion ratio {stat["ratio"]}')
    if arg.wandb:
        agent.logger.push_loss()
        agent.logger.push_stat(count, stat, test=False, commit=True)

def test(count, commit):
    stats, frames = [], []
    for _ in range(arg.test_epi):
        stat, frame, policy = play(train=False)
        stats.append(stat)
        if arg.gif:
            frames.append((frame, policy))

    log_stat = dict()
    for key in stats[0].keys():
        log_stat[key] = sum([stat[key] for stat in stats]) / len(stats)
    print(f'Test episode {count}: done {log_stat["done"]}, makespan {log_stat["makespan"]}, reward {log_stat["reward"]},' 
          f' completion ratio {log_stat["ratio"]}')

    if arg.wandb:
        agent.logger.push_stat(count, log_stat, test=True, commit=commit)
    if arg.gif:
        with open('frames.pkl', 'wb') as f:
            pk.dump(frames, f)

def run():
    epi_count = 0

    while epi_count < arg.episode:
        # Test
        if epi_count % arg.test_freq == 0 and arg.test_epi > 0:
            test(epi_count, commit=False)
        if arg.test:
            return

        # Train
        epi_count += 1
        train(epi_count)

        # Save
        if arg.save_freq > 0 and epi_count % arg.save_freq == 0:
            if not os.path.exists(arg.save_dir):
                os.makedirs(arg.save_dir)
            path = f'{arg.save_dir}{arg.name}-{epi_count / arg.save_freq}.pt'
            torch.save(agent.model.state_dict(), path)
            print(f'Saved to {path}')

    # Final test
    if arg.test_epi > 0:
        test(epi_count, commit=True)


if __name__ == '__main__':
    arg = config.get_parser().parse_args()
    arg = config.process(arg)
    if torch.cuda.is_available():
        arg.device = torch.device('cuda')
    else:
        arg.device = torch.device('cpu')
    print(f'Using device {arg.device}')

    env = lego.GridWorld(arg)
    agent = agent.A2C(arg, env)

    if arg.wandb:
        config = dict(
            w=arg.w, h=arg.h, map=arg.map, cost=arg.cost,
            epi_len=arg.epi_len, train_step=arg.train_step,
            lr=arg.lr, gamma=arg.gamma, clip=arg.clip, entropy=arg.entropy,

            translate=arg.translate, radiation=arg.R, trap=arg.T,
        )
        wandb.init(entity='macc', project=arg.project, group=arg.group, name=arg.name, config=config)
        wandb.define_metric('train')
        wandb.define_metric('train/*', step_metric='train')
        wandb.define_metric('test')
        wandb.define_metric('test/*', step_metric='test')

    run()
    # cProfile.run('env.reset()', sort='tottime')

    print('Done!')
