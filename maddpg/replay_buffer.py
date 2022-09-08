# -*- coding: utf-8 -*-
# @Time : 2022/9/4 11:16
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : replay_buffer.py
# @Project : maddpg-review
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, state_dim, agent_obs_dims, agent_actions_dims, n_agents, batch_size, agents):
        '''
        初始化经验回放缓冲
        :param max_size: 经验回放的最大缓冲
        :param state_dims: 联合状态=环境直接返回 或者 每个agent的观测值的组合 维度
        :param agent_obs_dims: 每个agent观测值的维度组合
        :param agent_actions_dims:  每个agent动作维度组合
        :param n_agents: agent的个数
        :param batch_size: 采样批次大小
        :param agents: agents的名字
        '''
        self.agents = agents
        self.mem_size = max_size
        # 存储指针用于定位
        self.mem_cntr = 0
        # 当前有几个agent
        self.n_agents = n_agents
        self.batch_size = batch_size
        # agent的动作空间
        self.agent_actions_dims = agent_actions_dims
        # agent中actor的输入维度，也就是agent的观测值维度
        self.agent_obs_dims = agent_obs_dims
        # 整体的状态，包括各个agent的观测值
        self.state_memory = np.zeros((self.mem_size, state_dim))
        # 整体的下一个状态，同样包括各个agent的观测值
        self.state_next_memory = np.zeros((self.mem_size, state_dim))
        # 每个agent的reward
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        # 每个agent的中止状态
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self._init_actor_memory_()

    def _init_actor_memory_(self):
        '''
        为了方便获得每个agent各自的动作状态转移，所以为每个agent设置了存储缓存
        :return:
        '''
        # 用来存储每个agent自己的状态转移
        self.agents_state_memory = []
        self.agents_new_state_memory = []
        self.agents_action_memory = []
        # add three agent buffer
        for i in range(self.n_agents):
            self.agents_state_memory.append(
                np.zeros((self.mem_size, self.agent_obs_dims[i]))
            )
            self.agents_new_state_memory.append(
                np.zeros((self.mem_size, self.agent_obs_dims[i]))
            )
            self.agents_action_memory.append(
                np.zeros((self.mem_size, self.agent_actions_dims[i]))
            )

    def push(self, obs : dict, state : list, actions : dict, rewards : dict, obs_next : dict, state_next : list, dones : dict):
        '''
        存放经验缓冲
        :param obs: 各个agent的观测值
        :param state: 当前联合状态
        :param actions: 联合动作
        :param rewards: 联合奖励
        :param obs_next: 下一个联合观测值
        :param state_next: 下一个联合状态
        :param dones: 是否结束
        :return:
        '''
        index = self.mem_cntr % self.mem_size
        rewards_collector = []
        dones_collector = []
        for agent_idx in range(self.n_agents):
            self.agents_state_memory[agent_idx][index] = obs.get(self.agents[agent_idx])
            self.agents_new_state_memory[agent_idx][index] = obs_next.get(self.agents[agent_idx])
            self.agents_action_memory[agent_idx][index] = actions.get(self.agents[agent_idx])
            rewards_collector.append(rewards.get(self.agents[agent_idx]))
            dones_collector.append(dones.get(self.agents[agent_idx]))
        self.reward_memory[index] = rewards_collector
        self.terminal_memory[index] = dones_collector
        self.state_memory[index] = state
        self.state_next_memory[index] = state_next
        self.mem_cntr += 1

    def sample_buffer(self):
        '''
        按照批次大小进行采样
        :return: 返回采样数据
        '''
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_next = self.state_next_memory[batch]
        terminal = self.terminal_memory[batch]

        agent_states = []
        agent_next_states = []
        agent_actions = []

        for agent_idx in range(self.n_agents):
            agent_states.append(self.agents_state_memory[agent_idx][batch])
            agent_next_states.append(self.agents_new_state_memory[agent_idx][batch])
            agent_actions.append(self.agents_action_memory[agent_idx][batch])

        return states, states_next, agent_states, agent_actions, rewards, agent_next_states, terminal


    def ready(self):
        '''
        返回当前经验缓冲区是否有足够的数据进行采样
        :return: True / False
        '''
        return self.mem_cntr >= self.batch_size

















