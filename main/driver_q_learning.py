import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from curve import HilbertCurve, ZCurve
from main.agent import Agent
from main.environment import CurveEnvironment

CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 3
DATA_SIZE = 15
BATCH_SIZE = 16
MAX_STEP = 200
CAPACITY = 10000
GAMMA = 0.99  # 시간 할인율
LEARNING_RATE = 1e-3
USE_RNN = False


class QLDriver:
    def __init__(self, dimension, order, data_size, learning_rate, capacity, batch_size, use_rnn=False):
        self.use_rnn = use_rnn

        self.env = CurveEnvironment(order=order, dim=dimension, data_size=data_size, life=20)
        self.agent = Agent(
            num_states=2 ** (dimension * order) * 3,
            num_actions=2 ** (dimension * order),
            network_type='dqn',
            learning_rate=learning_rate,
            capacity=capacity,
            batch_size=batch_size,
            use_rnn=use_rnn
        )

    def convert_state(self, inp_state):
        if self.use_rnn:
            torch_state = torch.tensor(inp_state, dtype=torch.float32).view(1, -1, self.env.dim + 1)
        else:
            torch_state = torch.tensor(inp_state, dtype=torch.float32).view(1, -1)
        return torch_state

    def run(self, max_episode=5000, max_step=1000, target_step=500, span=10):
        cost_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 cost
        reward_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 reward
        global_step = 0  # behavior net 를 업데이트하는 기준, 일정 steps 마다 behavior net를 target net로 업데이트 해준다.
        for ep in range(1, max_episode + 1):  # 최대 에피소드 수만큼 반복
            obs = self.env.reset()
            self.agent.replay_memory.reset()

            state = self.convert_state(obs)
            mean_cost = 0
            mean_reward = 0
            for step in range(1, max_step + 1):
                action = self.agent.get_action(state, ep)
                next_obs, reward, done, infos = self.env.step(action)

                next_state = self.convert_state(next_obs)
                # state, action, reward, next_state, done
                self.agent.replay_memory.memorize(state, action, next_state, reward, done)
                self.agent.update()

                state = next_state

                if done:
                    break

                global_step += 1
                if global_step % target_step == 0:
                    self.agent.update_network(global_step, verbose=False)

                # 추가 정보
                mean_cost = mean_cost + 1 / step * (infos['cost'] - mean_cost)
                mean_reward = mean_reward + 1 / step * (reward - mean_reward)

            cost_list[ep % len(cost_list)] = mean_cost
            reward_list[ep % len(reward_list)] = mean_reward

            if ep % span == 0:
                print(f'episode {ep} is over.')
                print('Average of the cost in the {} episodes : {:.3f} And Reward : {:.3f}'
                      .format(span, np.mean(cost_list), np.mean(reward_list)))


if __name__ == '__main__':
    np.random.seed(210)

    driver = QLDriver(dimension=DIM, order=ORDER, data_size=DATA_SIZE, learning_rate=LEARNING_RATE,
                      capacity=CAPACITY, batch_size=BATCH_SIZE, use_rnn=USE_RNN)
    driver.run(max_episode=5000, max_step=1000, target_step=500)
