import torch
import numpy as np
import wandb

class Buffer:
    def __init__(self, arg):
        self.device = arg.device
        self.data = dict()

        self.key = ['ob', 'action', 'reward', 'next_ob', 'done', 'valid']
        self.float_key = ['ob', 'next_ob', 'valid']

    def reset(self):
        self.data = dict()
        for key in self.key:
            self.data[key] = []

    def get_trajectory(self):
        traj = dict()
        for key in self.key:
            if key in self.float_key:
                traj[key] = torch.as_tensor(np.array(self.data[key]), device=self.device, dtype=torch.float32)
            else:
                traj[key] = torch.as_tensor(self.data[key], device=self.device)
        return traj


class Logger:
    def __init__(self):
        self.data = dict()
        self.loss_key = ['loss', 'value_loss', 'policy_loss', 'entropy']
        self.stat_key = ['reward']

    def reset(self):
        self.data = dict()
        for key in self.loss_key + self.stat_key:
            self.data[key] = []

    def log_loss(self, loss, value_loss, policy_loss, entropy, step):
        self.data['loss'].append(loss / step)
        self.data['value_loss'].append(value_loss / step)
        self.data['policy_loss'].append(policy_loss / step)
        self.data['entropy'].append(entropy / step)

    def log_stat(self, reward):
        self.data['reward'].append(reward)

    def push_loss(self):
        log_dict = dict()
        for key in self.loss_key:
            if len(self.data[key]) > 0:
                log_dict[key] = sum(self.data[key]) / len(self.data[key])
        wandb.log(log_dict, commit=False)

    def push_stat(self, count, status, test=False, commit=True):
        log_dict = dict()
        if test:
            prefix = 'test/'
            metric = 'test'
        else:
            prefix = 'train/'
            metric = 'train'
        log_dict[metric] = count
        for key in status.keys():
            log_dict[prefix + key] = status[key]
        wandb.log(log_dict, commit=commit)