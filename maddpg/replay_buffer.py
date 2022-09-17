# -*- coding: utf-8 -*-
# @Time : 2022/9/4 11:16
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : replay_buffer.py
# @Project : maddpg-review
import numpy as np

class ReplayBuffer():
    # 这个replaybuffer的性能很差，初始化的时候是先申请对应大小的空间，这样就会导致buffer特别大的时候初始化非常慢
    # 所以后面再设计buffer的时候一定要采用增量式添加
    def __init__(self, max_size, batch_size, env):
        '''
        初始化经验回放缓冲
        :param max_size: 经验回放的最大缓冲
        :param batch_size: 采样批次大小
        '''
        self.mem_size = max_size
        # 存储指针用于定位
        self.mem_cntr = 0
        self.env = env
        self.batch_size = batch_size
        # 整体的状态，包括各个agent的观测值
        self.state_memory = np.array([[0.0] * len(self.env.state()) for i in range(self.mem_size)])
        # 整体的下一个状态，同样包括各个agent的观测值
        self.state_next_memory = np.array([[0.0] * len(self.env.state()) for i in range(self.mem_size)])
        # 每个agent的reward装的是字典
        self.reward_memory = np.array([{} for _ in range(self.mem_size)])
        # 每个agent的中止状态装的也是字典
        self.terminal_memory = np.array([{} for _ in range(self.mem_size)])
        # 接下来的空间用来存储agent各自的观测值和状态转移
        self.agents_obs_memory = np.array([{} for _ in range(self.mem_size)])
        self.agents_action_meomory = np.array([{} for _ in range(self.mem_size)])
        self.agents_obs_next_memory = np.array([{} for _ in range(self.mem_size)])


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
        self.state_memory[index] = state
        self.state_next_memory[index] = state_next
        self.reward_memory[index] = rewards
        self.terminal_memory[index] = dones
        self.agents_obs_memory[index] = obs
        self.agents_action_meomory[index] = actions
        self.agents_obs_next_memory[index] = obs_next
        self.mem_cntr += 1

    def sample_buffer(self):
        '''
        按照批次大小进行采样
        :return: 返回采样数据
        '''
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        state = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        state_next = self.state_next_memory[batch]
        terminals = self.terminal_memory[batch]

        agents_obs=self.agents_obs_memory[batch]
        agents_actions = self.agents_action_meomory[batch]
        agent_obs_next = self.agents_obs_next_memory[batch]

        return state, state_next, agents_obs, agents_actions, rewards, agent_obs_next, terminals


    def ready(self):
        '''
        返回当前经验缓冲区是否有足够的数据进行采样
        :return: True / False
        '''
        return self.mem_cntr >= self.batch_size

















