import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InputLayer(nn.Module):
    def __init__(self, n_in, n_out, n_features=3, use_rnn=False):
        """
        입력 layer. 신경망의 state를 가장 처음으로 입력받는 레이어다. 
        note) LSTM의 sequence feature는 3으로 hardcoded 되어 있음
        :param n_in: 
        :param n_out: 
        :param use_rnn: LSTM 사용 유무
        """
        super(InputLayer, self).__init__()
        self.use_rnn = use_rnn
        self.n_features = n_features

        if use_rnn:
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_out, batch_first=True)
        else:
            self.fc = nn.Linear(in_features=n_in, out_features=n_out)

    def forward(self, x, *args):
        if self.use_rnn:
            output, _ = self.lstm(x)  # hidden state는 사용하지 않음
            output = output[:, -1:, :]
            output = output.squeeze(dim=1)
        else:
            output = self.fc(x)
        return output


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

    def __init__(self, n_in, n_out, n_mid=None, use_rnn=False):
        super(ActorCritic, self).__init__()
        n_mid = n_mid or (n_in + n_out) // 2

        self.input_layer = InputLayer(n_in=n_in, n_out=n_mid, use_rnn=use_rnn)
        self.first_actor = Actor(n_in=n_mid, n_mid=n_mid, n_out=n_out)
        self.second_actor = Actor(n_in=n_mid, n_mid=n_mid, n_out=n_out)
        self.critic = Critic(n_in=n_mid, n_mid=n_mid, n_out=1)

    def forward(self, x):
        """
        forward propagation
        :param x: input state
        :return:
        """
        x = torch.relu(self.input_layer(x))
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

    def __init__(self, input_size, output_size, hidden_size=None, use_rnn=False):
        super(DQN, self).__init__()
        m_mid = hidden_size or ((input_size + output_size) // 2)
        self.action_space = output_size

        self.input_layer = InputLayer(n_in=input_size, n_out=m_mid, use_rnn=use_rnn)
        self.fc1_1 = nn.Linear(m_mid, m_mid)
        self.fc1_2 = nn.Linear(m_mid, output_size)
        self.fc2_1 = nn.Linear(m_mid, m_mid)
        self.fc2_2 = nn.Linear(m_mid, output_size)

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
                actions = np.array(list(map(lambda x: torch.argmax(x).item(), (a1, a2))), dtype=np.int)
        else:  # random selection
            actions = np.random.choice(self.action_space, size=(2,))
        return actions


class PolicyGradient(nn.Module):
    """
        REINFORCE with Entropy (Monte Carlo)
    """

    def __init__(self, input_size, output_size, hidden_size=None, use_rnn=False):
        super(PolicyGradient, self).__init__()
        hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.first = InputLayer(n_in=input_size, n_out=hidden_size, use_rnn=use_rnn)
        self.first_add = nn.Linear(hidden_size, hidden_size)
        self.first_out = nn.Linear(hidden_size, output_size)
        self.second = nn.Linear(hidden_size, hidden_size)
        self.second_add = nn.Linear(hidden_size, hidden_size)
        self.second_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = torch.relu(self.first(x))
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
