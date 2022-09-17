# -*- coding: utf-8 -*-
# @Time : 2022/9/8 10:33
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : agent.py
# @Project : maddpg-review
import torch
from maddpg.networks import ActorNetwork, CriticNetwork
import numpy as np


class Agent:
    def __init__(self, agent_obs_dims, state_dims, n_actions, n_agents, agent_name, save_dir, device, alpha=0.01,
                 beta=0.01, fc1=64, fc2=64,
                 gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = agent_name
        self.save_dir = save_dir
        self.device = device
        self.actor = ActorNetwork(alpha, agent_obs_dims, fc1, fc2, n_actions, device=self.device , save_dir=self.save_dir +agent_name +'_actor.pth')
        self.target_actor = ActorNetwork(alpha, agent_obs_dims, fc1, fc2, n_actions, device=self.device ,save_dir=self.save_dir + agent_name + 'target_actor.pth')

        self.critic = CriticNetwork(beta, state_dims, fc1, fc2, n_agents, n_actions, device=self.device ,save_dir=self.save_dir + agent_name + 'critic.pth')
        self.target_critic = CriticNetwork(beta, state_dims, fc1, fc2, n_agents, n_actions, device=self.device ,save_dir=self.save_dir + agent_name  + 'target_critic.pth')
        # 根据参数可以调整为软更新或者直接更新,tau=1就是硬更新=0不更新
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.actor(obs)
        if self.device == 'cuda':
            return action.detach().cpu().numpy()
        else:
            return action.detach().numpy()

    def update_network_parameters(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) +
                param.data * tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) +
                param.data * tau
            )

    def save_model(self):
        self.actor.save_actor_model()
        self.target_actor.save_actor_model()
        self.critic.save_critic_model()
        self.target_critic.save_critic_model()

    def load_model(self):
        self.actor.load_actor_model()
        self.target_actor.load_actor_model()
        self.critic.load_critic_model()
        self.target_critic.load_critic_model()
