import torch
import numpy as np
import random

from torch import optim
from rl_network import *


class Agent:
    """
        학습 에이전트
        네트워크 모델을 받고, 학습을 수행함
    """

    def __init__(self, num_states, num_actions, network_type, learning_rate,
                 gamma=0.99, capacity=10000, batch_size=16):
        self.gamma = gamma
        self.network_type = network_type
        # 신경망 생성
        try:
            if network_type == 'policy_gradient':  # policy gradient
                self.network = PolicyGradient(num_states, num_actions)
            elif network_type == 'actor_critic':
                self.network = ActorCritic(n_in=num_states, n_out=num_actions)
                self.large_i = 1  # for update actor
            elif network_type == 'dqn':
                self.network = DQN(input_size=num_states, output_size=num_actions)
                self.replay_memory = ReplayMemory(capacity, batch_size)
            else:
                raise Exception('invalid network type {}'.format(network_type))
        except Exception as e:
            print(e)

        # optimizer 생성
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def update(self, history=None):
        if self.network_type == 'policy_gradient':
            assert history is not None

            curr_r = 0
            loss = torch.zeros(1, 1)
            for i in reversed(range(len(history))):
                log_prob, reward, entropy = history[i]
                curr_r = self.gamma * curr_r + reward
                for lgp, etp in zip(log_prob, entropy):
                    loss -= torch.sum(((curr_r * lgp) + (1e-3 * etp)))
            loss /= len(history)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.network_type == 'actor_critic':
            assert history is not None

            loss = torch.zeros(1, 1)
            log_prob, reward, entropy, value, next_state, done = history[-1]
            with torch.no_grad():
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)
                    next_value, _, _ = self.network(next_state)
                else:
                    next_value = 0
                delta = reward + self.gamma * next_value - value
            for lpb, etp in zip(log_prob, entropy):
                loss -= (delta * value + (self.large_i * lpb) + (1e-3 * etp))
            self.large_i *= self.gamma

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.network_type == 'DQN':
            loss = torch.zeros(1, 1)

            # 저장된 transition 수가 mini_batch 크기보다 작으면 아무 것도 안함
            if len(self.replay_memory) < self.replay_memory.batch_size:
                return

            # 미니배치 생성
            transitions = self.memory.sample()


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
            loss = loss_1 + loss_2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            raise Exception('Unknown network type update')

    def get_action(self, state, *args):
        outputs = self.network.get_action(state, *args)
        return outputs


class ReplayMemory:
    """
    DQN에서 사용할 replay-memory
    """
    def __init__(self, capacity, batch_size):
        self.capacity = capacity  # 메모리 최대 저장 건수
        self.memory = []  # 실제 transition 을 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수
        self.batch_size = batch_size

    def memorize(self, state, action, state_next, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, state_next, reward, done))
        else:
            self.memory[self.index] = (state, action, state_next, reward, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self):
        """
        batch_size 개수만큼 무작위로 저장된 transition 호출
        """
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


def discount_rewards(rewards, gamma):
    """
        take 1D float array of rewards and compute discounted reward
    """
    reward_shape = rewards.shape
    if len(reward_shape) == 1:
        discounted_r = np.zeros(shape=(*reward_shape, 1), dtype=np.float)
    else:
        discounted_r = np.zeros(shape=reward_shape, dtype=np.float)
    running_add = 0

    for t in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r
