import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_in, n_out, n_mid=None):
        super(Actor, self).__init__()
        n_mid = n_mid or (n_in + n_out) // 2
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        act = torch.softmax(self.actor(h2), dim=-1)  # 행동 계산
        return act


class Critic(nn.Module):
    def __init__(self, n_in, n_out, n_mid=None):
        super(Critic, self).__init__()
        n_mid = n_mid or (n_in + n_out) // 2
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.critic = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        critic = self.critic(h2)  # 상태가치 계산
        return critic


class ActorCritic(nn.Module):
    """
        One-step Actor-Critic
    """

    def __init__(self, n_in, n_out, n_mid=None):
        super(ActorCritic, self).__init__()
        n_mid = n_mid or (n_in + n_out) // 2
        self.input_layer = nn.Linear(n_in, n_mid)
        self.first_actor = Actor(n_in=n_mid, n_mid=n_mid, n_out=n_out)
        self.second_actor = Actor(n_in=n_mid, n_mid=n_mid, n_out=n_out)
        self.critic = Critic(n_in=n_mid, n_mid=n_mid, n_out=1)

    def forward(self, x):
        """
        forward propagation
        :param x: input state
        :return:
        """
        x = self.input_layer(x)
        a1 = self.first_actor(x)
        a2 = self.second_actor(x)
        cr = self.critic(x)

        return cr, a1, a2

    def get_action(self, x, *args):
        """상태 x로부터 행동을 확률적으로 결정"""
        value, ac1, ac2 = self(x)

        actions = []
        log_prob = []
        entropy = []
        for prob in [ac1, ac2]:
            selected_action = torch.multinomial(prob, num_samples=1)
            act_prob = torch.gather(prob, dim=1, index=selected_action)
            log_act_prob = torch.log(act_prob)
            act_entropy = -torch.sum((act_prob * log_act_prob))

            actions.append(selected_action.item())
            log_prob.append(log_act_prob)
            entropy.append(act_entropy)
        return actions, log_prob, entropy, value


class DQN(nn.Module):
    """
        DQN with replay buffer and double network
    """

    def __init__(self, input_size, output_size, hidden_size=None, lstm=False):
        super(DQN, self).__init__()
        m_in = hidden_size or ((input_size + output_size) // 2)
        self.action_space = output_size

        self.input_layer = nn.Linear(input_size, m_in)
        self.fc1_1 = nn.Linear(m_in, m_in)
        self.fc1_2 = nn.Linear(m_in, output_size)
        self.fc2_1 = nn.Linear(m_in, m_in)
        self.fc2_2 = nn.Linear(m_in, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        h_x_1 = torch.relu(self.fc1_1(x))
        v_1 = self.fc1_2(h_x_1)
        h_x_2 = torch.relu(self.fc2_1(x))
        v_2 = self.fc2_2(h_x_2)
        return v_1, v_2

    def get_action(self, state, *args):
        """
            epsilon-greedy selection
            :param state:
            :param args: args[0]: the number of episode
            :return:
        """
        episode_num = args[0]
        eps = 0.5 * (1 / np.log2(episode_num + 1 + 1e-7))

        if eps < np.random.uniform(0, 1):
            with torch.no_grad():  # greedy selection
                a1, a2 = self(state)
                actions = list(map(lambda x: x.item(), (a1, a2)))
        else:       # random selection
            actions = np.random.choice(self.action_space, size=(1, 2))
        return actions


class PolicyGradient(nn.Module):
    """
        REINFORCE with Entropy (Monte Carlo)
    """

    def __init__(self, input_size, output_size, hidden_size=None, lstm=False):
        super(PolicyGradient, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)
        if lstm is False:
            self.first = nn.Linear(input_size, self.hidden_size)
        else:
            self.first = nn.LSTM(input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.first_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.first_out = nn.Linear(self.hidden_size, output_size)
        self.second = nn.Linear(self.hidden_size, self.hidden_size)
        self.second_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.second_out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_val):
        output = torch.relu(self.first(input_val))
        output = torch.relu(self.first_add(output))
        first = self.first_out(output)
        first_output = torch.softmax(first, dim=-1)

        output = torch.relu(self.second(output))
        second = torch.relu(self.second_add(output))
        second = self.second_out(second)
        second_output = torch.softmax(second, dim=-1)
        return first_output, second_output

    def get_action(self, state, *args):
        action = []
        log_prob = []
        entropy = []
        for prob in self(state):
            act = torch.multinomial(prob, num_samples=1)
            act_prob = torch.gather(prob, dim=1, index=act)
            log_act_prob = torch.log(act_prob)
            act_entropy = -torch.sum((act_prob * log_act_prob))

            action.append(act.item())
            log_prob.append(log_act_prob)
            entropy.append(act_entropy)

        return action, log_prob, entropy
