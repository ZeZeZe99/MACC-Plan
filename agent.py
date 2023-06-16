import torch
import numpy as np

from network import ACNet
from util import Buffer, Logger

class Agent:
    def __init__(self, arg, env):

        self.device = arg.device
        self.env = env
        self.num_action = arg.num_action

        '''Training'''
        self.gamma = arg.gamma
        self.entropy = arg.entropy

        self.logger = Logger()
        self.buffer = Buffer(arg)

    def reset(self):
        self.buffer.reset()
        self.logger.reset()

    def store_transition(self, ob, action, reward, next_ob, done, valid):
        self.buffer.data['ob'].append(ob)
        self.buffer.data['action'].append(action)
        self.buffer.data['reward'].append(reward)
        self.buffer.data['next_ob'].append(next_ob)
        self.buffer.data['done'].append(done)
        self.buffer.data['valid'].append(valid)

class A2C(Agent):
    def __init__(self, arg, env):
        super(A2C, self).__init__(arg, env)

        self.model = ACNet(arg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=arg.lr)

    def bootstrap(self, next_ob, done):
        if done:
            next_value = 0
        else:
            next_value = self.model(next_ob)[1].detach()
        return next_value

    def discount_cumsum(self, gamma, values):
        discount = gamma ** torch.arange(values.shape[-1], device=self.device)
        v = values * discount
        v = v.flip([-1]).cumsum(-1).flip([-1]) / discount
        return v

    def calculate_discounted_reward(self, reward, boot_v):
        reward = torch.cat((reward, boot_v), dim=-1)
        discounted_r = self.discount_cumsum(self.gamma, reward)[:-1]
        return discounted_r

    def calculate_GAE(self, reward, value, boot_v):
        value = torch.cat((value, boot_v), dim=-1)
        advantage = reward + self.gamma * value[1:] - value[:-1]
        gae = self.discount_cumsum(self.gamma, advantage)
        return gae

    def calculate_loss(self):
        traj = self.buffer.get_trajectory()

        logit, value = self.model(traj['ob'])
        value = value.squeeze(-1)
        mask = traj['valid']
        logit -= (1 - mask) * 1e10
        policy = torch.softmax(logit, dim=-1)

        boot_v = self.bootstrap(traj['next_ob'][-1], traj['done'][-1])
        discounted_r = self.calculate_discounted_reward(traj['reward'], boot_v)
        gae = self.calculate_GAE(traj['reward'], value.detach(), boot_v)

        '''Policy loss'''
        selected_pi = torch.gather(policy, dim=1, index=traj['action'].unsqueeze(-1)).squeeze(-1)
        policy_loss = - (torch.clip(selected_pi, 1e-10, 1.0).log() * gae).sum()

        '''Value loss'''
        value_loss = 0.5 * (discounted_r - value).pow(2).sum()

        '''Entropy loss'''
        entropy_loss = self.entropy * (policy * torch.clip(policy, 1e-10, 1.0).log()).sum()

        loss = policy_loss + value_loss + entropy_loss

        # Log loss
        self.logger.log_loss(loss, value_loss, policy_loss, entropy_loss, policy.shape[0])
        return loss

    def update(self):
        self.optimizer.zero_grad()
        loss = self.calculate_loss()
        loss.backward()
        self.optimizer.step()

    def select_action(self, ob, valid):
        with torch.no_grad():
            ob = torch.as_tensor(ob, device=self.device, dtype=torch.float32)
            logit = self.model(ob)[0].cpu()
        mask = valid.reshape(-1)
        logit -= (1 - mask) * 1e10  # Mask out invalid action
        policy = torch.softmax(logit, dim=-1)
        if policy.sum() > 0:  # If policy is valid, normalize it
            policy /= policy.sum()
            action = torch.multinomial(policy, num_samples=1).item()
        else:  # If policy is invalid, choose from valid action randomly
            policy = mask / mask.sum()
            action = np.random.choice(self.num_action, p=policy)
        return action, policy
