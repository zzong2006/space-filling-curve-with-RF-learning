from rl_network import *
import torch
import numpy as np
from torch import optim


class Agent:
    """
        학습 에이전트
        네트워크 모델을 받고, 학습을 수행함
    """

    def __init__(self, num_states, num_actions, network_type, learning_rate, gamma=0.99):
        self.gamma = gamma
        self.network_type = network_type
        # 신경망 생성
        try:
            if network_type == 'policy_gradient':  # policy gradient
                self.network = PolicyGradient(num_states, num_actions)
            elif network_type == 'actor_critic':
                self.network = ActorCritic(n_in= num_states, n_out=num_actions)
                self.large_i = 1  # for update actor
            else:
                raise Exception('invalid network type {}'.format(network_type))
        except Exception as e:
            print(e)

        # optimizer 생성
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def update(self, history):
        if self.network_type == 'policy_gradient':
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

        else:
            raise Exception('U')

    def get_action(self, state):
        outputs = self.network.get_action(state)
        return outputs


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
