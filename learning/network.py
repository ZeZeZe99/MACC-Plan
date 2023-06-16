import torch
import torch.nn as nn


class ACNet(nn.Module):
    def __init__(self, arg):
        super(ACNet, self).__init__()

        self.w = arg.w
        self.h = arg.h

        input_dim = arg.input_dim
        num_action = arg.num_action
        relu, linear = [], []

        '''
        CNN layers
        Convolution: w' = (w - k + 2p) / s + 1
        Convolution transpose: w' = (w - 1) * s - 2p + k
        '''
        if self.w == 8:
            self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=0)  # 6x6
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # 4x4
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)  # 2x2
            self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)  # 1x1
            relu += [self.conv1, self.conv2, self.conv3, self.conv4]
        else:
            raise NotImplementedError

        '''FC layers'''
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        relu += [self.fc1, self.fc2, self.fc3]

        '''A2C Network'''
        self.policy1 = nn.Linear(256, 128)
        self.policy2 = nn.Linear(128, num_action)
        self.value1 = nn.Linear(256, 128)
        self.value2 = nn.Linear(128, 1)
        relu += [self.policy1, self.value1]
        linear += [self.policy2, self.value2]

        '''Weight initialization'''
        for layer in relu:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in linear:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')

    def forward(self, ob):
        x = torch.relu(self.conv1(ob))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=-3)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        a = torch.relu(self.policy1(x))
        logit = self.policy2(a)
        c = torch.relu(self.value1(x))
        value = self.value2(c)

        return logit, value


