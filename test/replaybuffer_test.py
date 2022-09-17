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

        self.max_size = 1024
        self.batch_size = 2

        self.replay_buffer = ReplayBuffer(max_size=self.max_size, batch_size=self.batch_size, env=self.env)

    def test_sample_add(self):
        obs = self.env.reset()
        state = self.env.state()
        for step in range(self.max_cycles):
            actions = {agent: [0, 0, 1, 0, 0] for agent in self.env.agents}
            obs_next, rewards, dones, _ = self.env.step(actions)
            state_next = self.env.state()
            self.replay_buffer.push(obs=obs, state=state, actions=actions, rewards=rewards, obs_next=obs_next,
                                    state_next=state_next, dones=dones)
        state, state_next, agents_obs, agents_actions, rewards, agents_obs_next, terminals = self.replay_buffer.sample_buffer()
        reward = [r['agent_0'] for r in rewards]
        print(reward)
        self.env.close()






if __name__ == '__main__':
    suite = unittest.TestSuite()
    tests = [test_add()]
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

