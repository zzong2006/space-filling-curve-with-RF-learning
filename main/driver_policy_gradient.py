import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from curve import HilbertCurve, ZCurve
from utils import *
from environment import CurveEnvironment
from agent import Agent

CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 3
DATA_SIZE = 15
MAX_STEP = 200
GAMMA = 0.99  # 시간 할인율
LEARNING_RATE = 1e-3  # 학습률
USE_RNN = True


class PGDriver:
    def __init__(self, dimension, order, data_size, learning_rate, use_rnn=False):
        self.use_rnn = use_rnn

        self.env = CurveEnvironment(order=order, dim=dimension, data_size=data_size, life=20)
        self.agent = Agent(
            num_states=2 ** (dimension * order) * 3,
            num_actions=2 ** (dimension * order),
            network_type='policy_gradient',
            learning_rate=learning_rate,
            use_rnn=use_rnn
        )

    def convert_state(self, inp_state):
        if self.use_rnn:
            torch_state = torch.tensor(inp_state, dtype=torch.float32).view(1, -1, self.env.dim + 1)
        else:
            torch_state = torch.tensor(inp_state, dtype=torch.float32).view(1, -1)
        return torch_state

    def run(self, max_episode=5000, max_step=1000, span=10):
        """

        :param max_episode: 수행할 최대 episode
        :param max_step: 한 episode에서 수행할 수 있는 max_step
        :param span: 최근 span 의 평균 보상, 또는 cost ...
        :return:
        """
        cost_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 cost
        reward_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 reward

        for ep in range(1, max_episode + 1):  # 최대 에피소드 수만큼 반복
            obs = self.env.reset()

            state = self.convert_state(obs)
            mean_cost = 0
            mean_reward = 0
            ep_history = []

            for step in range(1, max_step + 1):
                action, log_prob, entropy = self.agent.get_action(state)
                next_obs, reward, done, infos = self.env.step(action)

                ep_history.append([log_prob, reward, entropy])

                state = self.convert_state(next_obs)

                if done:
                    self.agent.update(ep_history)
                    break

                # 추가 정보
                mean_cost = mean_cost + 1 / step * (infos['cost'] - mean_cost)
                mean_reward = mean_reward + 1 / step * (reward - mean_reward)

            cost_list[ep % len(cost_list)] = mean_cost
            reward_list[ep % len(reward_list)] = mean_reward

            if ep % span == 0:
                print(f'episode {ep} is over.')
                print('Average of the cost in the {} episodes : {:.3f} And Reward : {:.3f}'
                      .format(span, np.mean(cost_list), np.mean(reward_list)))


'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''

if __name__ == '__main__':
    np.random.seed(210)

    driver = PGDriver(dimension=DIM, order=ORDER, data_size=DATA_SIZE, learning_rate=LEARNING_RATE, use_rnn=USE_RNN)
    driver.run(max_episode=5000, max_step=1000)
