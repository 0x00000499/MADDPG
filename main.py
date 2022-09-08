# -*- coding: utf-8 -*-
# @Time : 2022/9/4 11:26
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : main.py
# @Project : maddpg-review
import sys,os
import torch

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)
import datetime
import argparse
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
from maddpg.replay_buffer import ReplayBuffer
def get_cfg():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    # env set
    parser.add_argument('--algo_name', default='MADDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='simple_adversary_v2', type=str, help="name of environment")
    # parser.add_argument('--agents_n', default=2, type=int, help="agents amount")
    # parser.add_argument('--continuous_actions', default=True, type=bool, help="continuous env or discrete")
    # parser.add_argument('--max_cycles', default=5000, type=int, help="max env cycles")
    # train test set
    parser.add_argument('--train_eps', default=300, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--buffer_size', default=8000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    return args


def make_env_with_buffer(cfg, seed = None, agents_n = 2, continuous_actions=True, max_cycles = 5000, buffer_size = 1024):
    env = simple_adversary_v2.parallel_env(N=agents_n, continuous_actions=continuous_actions, max_cycles = max_cycles)
    env.reset()
    if seed is not None:
        env.seed(seed)
    n_agents = env.num_agents
    agents = env.agents
    state_dim = env.state_space.shape[0]
    agent_obs_dims = []
    agent_action_dims = []
    batch_size = cfg.batch_size
    for agent_idx in range(n_agents):
        agent_obs_dims.append(env.observation_space(env.agents[agent_idx]).shape[0])
        agent_action_dims.append(env.action_space(env.agents[agent_idx]).shape[0])

    replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=state_dim,
                                      agent_obs_dims=agent_obs_dims, agent_actions_dims=agent_action_dims,
                                      n_agents=n_agents, batch_size=batch_size, agents=agents)
    return env, replay_buffer



def train(cfg):
    max_cycles = 500
    env, replay_buffer = make_env_with_buffer(cfg = cfg, max_cycles = max_cycles)
    obs = env.reset()
    state = env.state()
    for step in range(max_cycles):
        # action类型必须是np.float32! 第一位无操作，2-5位给定四个方向上的速度
        action = np.array([0,1,0,0,0],dtype=np.float32)
        actions = {agent: action for agent in env.agents}
        obs_next, rewards, dones, _ = env.step(actions)
        state_next = env.state()
        replay_buffer.push(obs=obs, state=state, actions=actions, rewards=rewards, obs_next=obs_next,
                                state_next=state_next, dones=dones)
        env.render()
    print(replay_buffer.sample_buffer())
    env.close()
    '''
    训练模型
    :return:
    '''
 # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    cfg = get_cfg()
    train(cfg)

