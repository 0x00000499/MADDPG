# -*- coding: utf-8 -*-
# @Time : 2022/9/4 11:26
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : replaybuffer_test.py
# @Project : maddpg-review
import unittest
import numpy as np
import run
from pettingzoo.mpe import simple_adversary_v2
from maddpg.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.max_cycles = 1000
        self.env = simple_adversary_v2.parallel_env(N=2, continuous_actions=True, max_cycles=self.max_cycles)
        self.env.reset()
        self.env.render(mode='human')

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.state_dim = self.env.state_space.shape[0]
        self.agent_obs_dims = []
        self.agent_action_dims = []
        self.max_size = 1024
        self.batch_size = 2
        for agent_idx in range(self.n_agents):
            self.agent_obs_dims.append(self.env.observation_space(self.env.agents[agent_idx]).shape[0])
            self.agent_action_dims.append(self.env.action_space(self.env.agents[agent_idx]).shape[0])

        self.replay_buffer = ReplayBuffer(max_size=self.max_size, state_dim=self.state_dim,
                                          agent_obs_dims=self.agent_obs_dims, agent_actions_dims=self.agent_action_dims,
                                          n_agents=self.n_agents, batch_size=self.batch_size, agents=self.agents)

    def test_sample_add(self):
        obs = self.env.reset()
        state = self.env.state()
        for step in range(self.max_cycles):
            actions = {agent: [1, 0, 0, 0, 0] for agent in self.env.agents}
            obs_next, rewards, dones, _ = self.env.step(actions)
            state_next = self.env.state()
            self.replay_buffer.push(obs=obs, state=state, actions=actions, rewards=rewards, obs_next=obs_next,
                                    state_next=state_next, dones=dones)
        print(self.replay_buffer.sample_buffer())
        self.env.close()






if __name__ == '__main__':
    suite = unittest.TestSuite()
    tests = [test_add()]
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

