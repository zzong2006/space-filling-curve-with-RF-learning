import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # 행동을 결정하는 부분이므로 출력 갯수는 행동의 가짓수
        self.critic = nn.Linear(n_mid, 1)  # 상태가치를 출력하는 부분이므로 출력 갯수는 1개

    def forward(self, x):
        """신경망 순전파 계산을 정의"""
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # 상태가치 계산
        actor_output = self.actor(h2)  # 행동 계산

        return critic_output, actor_output

    def act(self, x):
        """상태 x로부터 행동을 확률적으로 결정"""
        value, actor_output = self(x)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def get_value(self, x):
        """상태 x로부터 상태가치를 계산"""
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        """상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산"""
        value, actor_output = self(x)
        # print(actor_output.data)
        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions)  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


class PolicyGradient(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyGradient, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.first = nn.Linear(input_size, self.hidden_size)
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

    def get_action(self, state):
        with torch.no_grad():
            first, second = self.model(state.view(1, -1))
            a_dist = first.cpu().detach().numpy().reshape([-1])
            a = np.random.choice(a_dist, p=a_dist)
            a = np.argmax(a_dist == a)
            b_dist = second.cpu().detach().numpy().reshape([-1])
            b = np.random.choice(b_dist, p=b_dist)
            b = np.argmax(b_dist == b)

            return torch.cuda.FloatTensor([[a, b]])
