# -*- coding: utf-8 -*-
# @Time : 2022/9/4 11:26
# @Author : Gaoxinzhi
# @Email : dut_gaoxinzhi@qq.com
# @File : main.py
# @Project : maddpg-review
import sys, os
from tqdm import tqdm
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)
import datetime
import argparse
import numpy as np
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_v2
from maddpg.replay_buffer import ReplayBuffer
from maddpg.maddpg_algo import MADDPG
import matplotlib.pyplot as plt
import time

def get_cfg():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    # env set
    parser.add_argument('--algo_name', default='MADDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='simple_push', type=str, help="name of environment")
    parser.add_argument('--max_cycles', default=25, type=int, help="max cycle for episode")
    parser.add_argument('--continuous_actions', default=True, type=bool, help="continuous env or discrete")
    parser.add_argument('--agents_n', default=2, type=int, help="agents number")
    parser.add_argument('--best_score', default=-15, type=int, help="best score baseline for env")
    # train test set
    parser.add_argument('--train_eps', default=50000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=0.01, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=0.01, type=float, help="learning rate of actor")
    parser.add_argument('--buffer_size', default=1000000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--soft_tau', default=0.01, type=float)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--result_path', default=curr_path + "\\outputs\\" + 'results\\')
    parser.add_argument('--model_path', default=curr_path + "\\outputs\\" + 'models\\')  # path to save models
    parser.add_argument('--evaluate', default=True, type=bool, help="evaluate")
    args = parser.parse_args()
    return args

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切


def make_env(cfg, seed=None, agents_n=2, continuous_actions=True, max_cycles=25):
    env = simple_push_v2.parallel_env(continuous_actions=continuous_actions, max_cycles=max_cycles)
    env.reset()
    return env


def train(cfg):
    fig, ax = plt.subplots()
    eposide_record = []
    score_mean = []
    ############################
    model_path = cfg.model_path + cfg.env_name + "\\"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    env = make_env(cfg, agents_n=cfg.agents_n, continuous_actions=cfg.continuous_actions, max_cycles=cfg.max_cycles)
    replay_buffer = ReplayBuffer(max_size=cfg.buffer_size, batch_size=cfg.batch_size, env=env)
    maddpg_agents = MADDPG(env, cfg.device, fc1=cfg.hidden_dim, fc2=cfg.hidden_dim, alpha=cfg.actor_lr,
                           save_dir=model_path, beta=cfg.critic_lr, tau=cfg.soft_tau,gamma=cfg.gamma)
    noise = {}
    for a_n in env.agents:
        noise[a_n] = OUNoise(env.action_space(a_n))
    total_step = 0
    best_score = cfg.best_score
    score_history = []
    for eposide in tqdm(range(cfg.train_eps)):
        for a_n in env.agents:
            noise[a_n].reset()
        obs = env.reset()
        state = env.state()
        # 记录每个回合的成绩
        score = 0
        for step in range(cfg.max_cycles):
            # action类型必须是np.float32! 第一位无操作，2-5位给定四个方向上的速度
            actions = maddpg_agents.choose_actions(obs)
            for a_n in env.agents:
                actions[a_n] += np.random.rand(env.action_space(a_n).shape[0]) if eposide < 10000 else np.zeros(env.action_space(a_n).shape[0])
                actions[a_n] = np.clip(actions[a_n], env.action_space(a_n).low, env.action_space(a_n).high).astype(np.float32)
            obs_next, rewards, dones, _ = env.step(actions)
            if step == cfg.max_cycles - 1:
                for a_n in env.agents:
                    dones[a_n] = True
            state_next = env.state()
            replay_buffer.push(obs=obs, state=state, actions=actions, rewards=rewards, obs_next=obs_next,
                               state_next=state_next, dones=dones)
            obs = obs_next
            state = state_next
            total_step += 1
            if total_step % 100 == 0:
                maddpg_agents.learn(replay_buffer)
            score += np.sum(list(rewards.values()))
        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        if average_score > best_score and eposide > 0:
            best_score = average_score
            maddpg_agents.save_algo()
        if eposide % 500 == 0 and eposide > 0:
            eposide_record.append(eposide)
            score_mean.append(average_score)
            print("epoch", eposide, 'average score {:.1f}'.format(average_score),
                  'best score {:.1f}'.format(best_score))
    env.close()
    ############################
    ax.plot(eposide_record, score_mean)
    ax.set_xlabel('eposide')
    ax.set_ylabel('100 times average reward')
    ax.set_title('reward curve')
    res_path = cfg.result_path + cfg.env_name
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    fig.savefig(res_path + "\\result.jpg")
    fig.show()



def test(cfg):
    env = make_env(cfg)
    model_path = cfg.model_path + cfg.env_name + "\\"
    maddpg_agents = MADDPG(env, cfg.device, fc1=cfg.hidden_dim, fc2=cfg.hidden_dim, alpha=cfg.actor_lr,
                           save_dir=model_path, beta=cfg.critic_lr, tau=cfg.soft_tau, gamma=cfg.gamma)
    maddpg_agents.load_algo()
    for epoch in range(cfg.test_eps):
        obs = env.reset()
        for step in range(cfg.max_cycles):
            # action类型必须是np.float32! 第一位无操作，2-5位给定四个方向上的速度
            # 这里注意在测试的时候一定要传入正确的obs，一开始没有更新obs导致我查了好久的bug查不出来，一直以为是训练的问题
            actions = maddpg_agents.choose_actions(obs)
            obs, rewards, dones, _ = env.step(actions)
            env.render()
            time.sleep(0.05)
    env.close()


if __name__ == '__main__':
    cfg = get_cfg()
    if not cfg.evaluate:
        train(cfg)
    else:
        test(cfg)
