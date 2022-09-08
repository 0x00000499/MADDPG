# -*- coding: utf-8 -*-
# @Time : 2022/9/7 10:51
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : networks.py
# @Project : maddpg-review

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dims, fc1_dims, fc2_dims, n_agents, n_actions, save_dir):
        super(CriticNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dims + n_agents * n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(fc2_dims),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.save_dir = save_dir

    def forward(self, state, action):
        return self.linear_relu_stack(torch.cat([state, action], dim=0))

    def save_critic_model(self):
        torch.save(self.state_dict(), self.save_dir)

    def load_critic_model(self):
        self.load_state_dict(torch.load(self.save_dir))
        self.eval()


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dims, fc1_dims, fc2_dims, n_actions, save_dir):
        super(ActorNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(fc2_dims),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=0)
        )
        self.save_dir = save_dir
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        return self.linear_relu_stack(state)

    def save_actor_mode(self):
        torch.save(self.state_dict(), self.save_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.save_dir))
        self.eval()


if __name__ == '__main__':
    actor = ActorNetwork(alpha=0.01, state_dims=10, fc1_dims=64, fc2_dims=64, n_actions=5, save_dir="cao")

    input = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pi = actor.forward(input)
    print(pi)