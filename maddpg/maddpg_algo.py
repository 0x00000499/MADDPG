# -*- coding: utf-8 -*-
# @Time : 2022/9/8 11:03
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : maddpg_algo.py
# @Project : maddpg-review


import torch
import torch.nn.functional as F
from maddpg.agent import Agent
from maddpg.replay_buffer import ReplayBuffer
import numpy as np

class MADDPG:
    def __init__(self, env, device, alpha=0.5, beta=0.5, fc1=64, fc2=64, gamma=0.99, tau=0.01,
                 save_dir=""):
        self.agents = {}
        self.env = env
        self.n_agents = len(self.env.agents)
        self.device = device
        self.tau = tau
        for name in self.env.agents:
            self.agents[name] = Agent(agent_obs_dims=self.env.observation_space(name).shape[0],
                                      state_dims=self.env.state_space.shape[0],
                                      n_actions=self.env.action_space(name).shape[0],
                                      n_agents=self.n_agents,
                                      agent_name=name,
                                      save_dir=save_dir,
                                      device=self.device,
                                      alpha=alpha,
                                      beta=beta,
                                      fc1=fc1,
                                      fc2=fc2,
                                      gamma=gamma,
                                      tau=tau)

    def save_algo(self):
        print("saving model")
        for agent in self.agents.values():
            agent.save_model()

    def load_algo(self):
        print("loading model")
        for agent in self.agents.values():
            agent.load_model()

    def choose_actions(self, agents_obs):
        actions = {}
        for agent_name, agent in self.agents.items():
            action = agent.choose_action(agents_obs[agent_name])
            action = action.astype(np.float32)
            action = np.clip(action, self.env.action_space(agent_name).low ,self.env.action_space(agent_name).high)
            actions[agent_name] = action
        return actions

    def learn(self, replay_buffer: ReplayBuffer):
        if not replay_buffer.ready():
            return
        state, state_next, agents_obs, agents_actions, rewards, agents_obs_next, dones = replay_buffer.sample_buffer()

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state_next = torch.tensor(state_next, dtype=torch.float, device=self.device)

        all_agents_actions = {}
        all_agents_next_actions = {}
        all_agents_actual_acitons = {}

        for agent_name, agent in self.agents.items():
            all_agents_actual_acitons[agent_name] = torch.tensor([a[agent_name] for a in agents_actions], device=self.device)

            agent_obs = torch.tensor([o[agent_name] for o in agents_obs], dtype=torch.float32, device=self.device)
            all_agents_actions[agent_name] = agent.actor(agent_obs)

            agent_obs_next = torch.tensor([o_n[agent_name] for o_n in agents_obs_next], dtype=torch.float32, device=self.device)
            all_agents_next_actions[agent_name] = agent.target_actor(agent_obs_next)

        actions = torch.cat([act for act in all_agents_actions.values()], dim=1).to(self.device)
        next_actions = torch.cat([act for act in all_agents_next_actions.values()], dim=1).to(self.device)
        actual_actions = torch.cat([act for act in all_agents_actual_acitons.values()], dim=1).to(self.device)
        actual_actions = actual_actions.to(torch.float32).detach()

        for agent_name, agent in self.agents.items():
            critic_value = agent.critic(state, actual_actions).flatten()
            critic_next_value = agent.target_critic(state_next, next_actions).flatten()
            reward = torch.tensor([r[agent_name] for r in rewards], device=self.device)
            done = [d[agent_name] for d in dones]
            target = reward + agent.gamma * critic_next_value * (1 - (1 if done else 0))
            # 在这里不能执行参数更新操作，因为上面的actions是三个agent全部的推断，所以在遍历agent的时候
            # 会多次对同一agent内的actor网络进行更新操作，会出现in-place... error
            critic_loss = F.mse_loss(target.detach(), critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            actor_loss = agent.critic(state, actions).flatten()
            actor_loss = -actor_loss.mean()
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

        for agent in self.agents.values():
            agent.actor.optimizer.step()
            agent.update_network_parameters(tau=self.tau)








