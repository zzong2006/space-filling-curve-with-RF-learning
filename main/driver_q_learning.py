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
BATCH_SIZE = 32
MAX_STEP = 200
CAPACITY = 10000
GAMMA = 0.99  # 시간 할인율
LEARNING_RATE = 1e-3

Transition = namedtuple('Trainsition', ('state', 'action', 'next_state', 'reward'))

'''
    SFC를 만드는 모델
'''


class SFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SFCNet, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.first = nn.Linear(input_size, self.hidden_size)
        self.first_relu = nn.ReLU()
        self.first_out = nn.Linear(self.hidden_size, output_size)
        self.second = nn.Linear(self.hidden_size, self.hidden_size)
        self.second_relu = nn.ReLU()
        self.second_out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input):
        output = self.first_relu(self.first(input))
        first_output = self.first_out(output)
        output = self.second_relu(self.second(output))
        second_output = self.second_out(output)
        return first_output, second_output


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 메모리 최대 저장 건수
        self.memory = []  # 실제 transition을 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수

    '''
    transition = (state, action, state_next, reward)을 메모리에 저장
    '''

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Transition이라는 namedtuple을 사용해 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    '''
    batch_size 개수만큼 무작위로 저장된 transition 호출
    '''

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    '''
    len 함수로 현재 저장된 transition 개수를 반환
    '''

    def __len__(self):
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = SFCNet(num_states, hidden_size, num_actions)
        if CUDA: self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        print(self.model)

    '''
    Experience Replay로 신경망의 결합 가중치 학습
    '''

    def replay(self):
        # 저장된 transition 수가 mini_batch 크기보다 작으면 아무 것도 안함
        if len(self.memory) < BATCH_SIZE:
            return

        # 미니배치 생성
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states_batch = torch.cat(batch.next_state)
        # 정답 신호로 사용할 Q(s_t, a_t)를 계산
        self.model.eval()
        # set_trace()
        first, second = self.model(state_batch)
        first_state_action_values = first.gather(1, action_batch[:, 0].view(-1, 1))
        second_state_action_values = second.gather(1, action_batch[:, 1].view(-1, 1))

        # max{Q(s_{t+1}, a)} 값을 계산
        first, second = self.model(next_states_batch)

        first_next_state_values = first.max(1)[0].detach()
        second_next_state_values = second.max(1)[0].detach()

        # 정답 신호로 사용할 Q(s_t, a_t)값을 Q 러닝으로 계산
        # SAV : State Action Values

        first_expected_SAV = reward_batch + GAMMA * first_next_state_values
        second_expected_SAV = reward_batch + GAMMA * second_next_state_values

        self.model.train()
        # smooth_l1_loss는 Huber 함수
        loss_1 = F.smooth_l1_loss(first_state_action_values, first_expected_SAV.unsqueeze(1))
        loss_2 = F.smooth_l1_loss(second_state_action_values, second_expected_SAV.unsqueeze(1))
        total_loss = loss_1 + loss_2
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        epsilon = 0.5 * (1 / np.log2(episode + 1 + 1e-7))

        if epsilon < np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                first, second = self.model(state.view(1, -1))
                return torch.cat((first, second)).max(1)[1].view(1, 2)
        else:
            action = np.random.choice(self.num_actions, size=(1, 2))
            action = torch.from_numpy(action).type(torch.cuda.LongTensor)
        return action


# class Agent:
#     def __init__(self, num_states, num_actions):
#         self.brain = Brain(num_states, num_actions)
#
#     def update_q_function(self):
#         self.brain.replay()
#
#     def get_action(self, state, step):
#         action = self.brain.decide_action(state, step)
#         return action
#
#     def memorize(self, state, action, state_next, reward):
#         self.brain.memory.push(state, action, state_next, reward)


class QLDriver:
    def __init__(self, dimension, order, data_size, learning_rate):
        self.env = CurveEnvironment(order=order, dim=dimension, data_size=data_size, life=20)
        self.agent = Agent(
            num_states=2 ** (dimension * order) * 3,
            num_actions=2 ** (dimension * order),
            network_type='q_learning',
            learning_rate=learning_rate
        )

    def run(self, max_episode=5000, max_step=1000, span=10):
        cost_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 cost
        reward_list = np.zeros(span)  # 에피소드 당 달성할 수 있는 평균 reward
        for ep in range(max_episode):  # 최대 에피소드 수만큼 반복
            obs = self.env.reset()

            state = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            mean_cost = 0
            mean_reward = 0
            for step in range(max_step):
                action = self.agent.get_action(state, ep)

                next_obs, reward, done, infos = self.env.step(action)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()

                state = torch.tensor(next_obs, dtype=torch.float32).view(1, -1)

                if done:
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


if __name__ == '__main__':
    np.random.seed(210)

    driver = QLDriver(dimension=DIM, order=ORDER, data_size=DATA_SIZE, learning_rate=LEARNING_RATE)
    driver.run(max_episode=5000, max_step=1000)
