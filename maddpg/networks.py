# -*- coding: utf-8 -*-
# @Time : 2022/9/7 10:51
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : networks.py
# @Project : maddpg-review

import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dims, fc1_dims, fc2_dims, n_agents, n_actions, device, save_dir):
        super(CriticNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dims + n_agents * n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.save_dir = save_dir
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        data_in = torch.cat([state, action], dim=1)
        return self.linear_relu_stack(data_in)

    def save_critic_model(self):
        torch.save(self.state_dict(), self.save_dir)

    def load_critic_model(self):
        self.load_state_dict(torch.load(self.save_dir))
        self.eval()


class ActorNetwork(nn.Module):
    def __init__(self, alpha, obs_dims, fc1_dims, fc2_dims, n_actions, device, save_dir):
        super(ActorNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Sigmoid()
        )
        self.save_dir = save_dir
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, obs):
        return self.linear_relu_stack(obs)

    def save_actor_model(self):
        torch.save(self.state_dict(), self.save_dir)

    def load_actor_model(self):
        self.load_state_dict(torch.load(self.save_dir))
        self.eval()


if __name__ == '__main__':
    net = ActorNetwork(0.01, 19, 64, 64, 5, 'cuda', '..\\outputs\\models\\agent_0_actor.pth')
    net.load_actor_model()
    print(net(torch.tensor([-0.13578, 0.01653, -0.77000, -1.26925, 0.25000, 0.75000, 0.25000, -0.77000, -1.26925, -0.09854, -0.61911,
               0.10000, 0.90000, 0.10000, 0.10000, 0.10000, 0.90000, -0.04686, -1.02172
               ], device='cuda')))
