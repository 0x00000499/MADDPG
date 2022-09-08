# -*- coding: utf-8 -*-
# @Time : 2022/9/8 10:12
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : network_test.py
# @Project : maddpg-review

import unittest
from maddpg.networks import *

class TestNetworks(unittest.TestCase):
    def setUp(self) -> None:
        self.actor = ActorNetwork(alpha=0.01, state_dims=10, fc1_dims=64, fc2_dims=64, n_actions=5, save_dir="cao")
        self.critic = CriticNetwork(beta=0.01, state_dims=10, fc1_dims=64, fc2_dims=64, n_agents= 3, n_actions=5, save_dir="shit")

    def test_forward(self):
        pass






















