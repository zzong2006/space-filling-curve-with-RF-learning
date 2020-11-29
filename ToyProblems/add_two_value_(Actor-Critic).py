

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import math
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

GAMMA = 0.99


class Environment:
    def __init__(self):
        self.a = 1.232
        self.b = 1.545
        self.prev = np.abs(self.a - self.b)

    def reset(self):
        self.a = 1.232
        self.b = 1.545
        self.prev = np.abs(self.a - self.b)
        return np.array([self.a, self.b, self.prev])

    def step(self, action):
        done = False
        if action == 0:
            self.a += 0.01
            self.b += 0.01
        if action == 1:
            self.a -= 0.01
            self.b -= 0.01
        if action == 2:
            self.a += 0.01
            self.b -= 0.01
        if action == 3:
            self.a -= 0.01
            self.b += 0.01
        curr = np.abs(self.a - self.b)

        if self.prev > curr:
            done = True
            reward = -1
        elif self.prev == curr:
            reward = 0
        else:
            reward = 1

        self.prev = curr
        return np.array([self.a, self.b, self.prev]), reward, done

    def modified_step(self, action):
        done = False

        if action[0] == 0:
            self.a += 0.001
        elif action[0] == 1:
            self.a -= 0.001
        elif action[0] == 2:
            self.a = 1
        elif action[0] == 3:
            self.a = 3

        if action[1] == 0:
            self.b += 0.001
        elif action[1] == 1:
            self.b -= 0.001
        elif action[1] == 2:
            self.b = 2
        elif action[1] == 3:
            self.b = 1

        curr = np.abs(self.a - self.b)

        if math.isclose(self.prev, curr, rel_tol=1e-5):
            reward = -0.05
        elif self.prev > curr:
            done = True
            # print('현재 |a-b|가 이전 |a-b| 보다 작음. 감점 !',self.a, self.b, self.prev, curr)
            reward = -5
        else:
            reward = 5

        self.prev = curr
        return np.array([self.a, self.b, self.prev]), reward, done


class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''

    def __init__(self, num_steps, num_processes, obs_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_size).to(DEVICE)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(DEVICE)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(DEVICE)
        self.actions = torch.zeros(num_steps, num_processes, 2).long().to(DEVICE)

        # 할인 총보상 저장
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(DEVICE)
        self.index = 0  # insert할 인덱스

    def insert(self, current_obs, action, reward, mask):
        '''현재 인덱스 위치에 transition을 저장'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 인덱스 값 업데이트

    def after_update(self):
        '''Advantage학습 단계만큼 단계가 진행되면 가장 새로운 transition을 index0에 저장'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantage 학습 범위 안의 각 단계에 대해 할인 총보상을 계산'''

        # 주의 : 5번째 단계부터 거슬러 올라오며 계산
        # 주의 : 5번째 단계가 Advantage1, 4번째 단계는 Advantage2가 됨
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    # Normalize reward to avoid a big variability in rewards
    mean = np.mean(discounted_r)
    std = np.std(discounted_r)
    if std == 0: std = 1
    normalized_discounted_r = (discounted_r - mean) / std
    return normalized_discounted_r


class ActorCriticAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCriticAgent, self).__init__()
        self.i = nn.Linear(input_size, hidden_size)
        self.h = nn.Linear(hidden_size, hidden_size)
        self.a1 = nn.Linear(hidden_size, output_size)
        self.a2 = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output = torch.relu(self.i(input))
        output = torch.relu(self.h(output))
        first_action = self.a1(output)
        second_action = self.a2(output)
        value = self.critic(output)

        return [first_action, second_action], value


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Set total number of episodes to train agent on.
total_episodes = 1000
max_ep = 999
update_frequency = 5

i = 0
total_reward = 0
accumulated_reward = np.array([])
total_length = []
rlist = []

lr = 1e-3
s_size = 3  # state
a_size = 4  # action
h_size = 32  # hidden
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_ADVANCED_STEP = 5
NUM_PROCESSES = 8
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.001
MAX_GRAD_NORM = 0.5

model = ActorCriticAgent(s_size, h_size, a_size)
model.apply(init_weights).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr)
env = np.array([Environment()] * NUM_PROCESSES)

done_np = np.zeros([NUM_PROCESSES, 1])
obs_np = np.zeros([NUM_PROCESSES, s_size])  # Numpy 배열
reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, s_size)  # rollouts 객체

s = np.array([env[k].reset() for k in range(NUM_PROCESSES)])
rollouts.observations[0].copy_(torch.from_numpy(s).float().to(DEVICE))

print('DEVICE (CUDA) : ', DEVICE)

while i < total_episodes:
    if i % 100 == 0:
        print(f'Start {i}th Episode ... ')

    running_reward = 0

    for j in range(NUM_ADVANCED_STEP):
        # 네트워크 출력에서 확률적으로 액션을 선택
        with torch.no_grad():
            action, value = model(rollouts.observations[j])
            a = (torch.softmax(action[0], dim=1)).multinomial(1).data
            b = (torch.softmax(action[1], dim=1)).multinomial(1).data
            selected_action = torch.cat((a, b), 1)

        for z in range(NUM_PROCESSES):
            obs_np[z], reward_np[z], done_np[z] = env[z].modified_step(
                [a[z], b[z]])  # Get our reward for taking an action
            if z == 0:
                total_reward += (reward_np[z].item())
                rlist.append(total_reward)
            if done_np[z] == True:
                # obs_np[z] = env[z].reset()
                if z == 0:
                    total_reward = 0
        accumulated_reward = np.append(accumulated_reward, reward_np)
        reward = torch.from_numpy(reward_np).float().to(DEVICE)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])

        current_obs = torch.from_numpy(obs_np).float().to(DEVICE)
        rollouts.insert(current_obs, selected_action.data, reward, masks)

    # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산
    with torch.no_grad():
        _, next_value = model(rollouts.observations[-1])
    rollouts.compute_returns(next_value)

    # 신경망 및 rollout 업데이트
    model.eval()
    obs_shape = rollouts.observations[:-1].size()
    actor_output, values = model(rollouts.observations[:-1].view(-1, s_size))
    log_probs_1 = F.log_softmax(actor_output[0], dim=1)
    log_probs_2 = F.log_softmax(actor_output[1], dim=1)

    action_log_probs_1 = log_probs_1.gather(1, rollouts.actions[:, :, 0].view(-1, 1))
    action_log_probs_2 = log_probs_2.gather(1, rollouts.actions[:, :, 1].view(-1, 1))

    probs_1 = F.softmax(actor_output[0], dim=1)
    probs_2 = F.softmax(actor_output[1], dim=1)

    # 엔트로피 H : action이 확률적으로 얼마나 퍼져 있는가? (비슷한 확률의 다중 액션 -> high, 단일 액션 -> low)
    entropy1 = -((log_probs_1 * probs_1)).sum(-1).mean()
    entropy2 = -((log_probs_2 * probs_2)).sum(-1).mean()

    values = values.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
    action_log_probs_1 = action_log_probs_1.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
    action_log_probs_2 = action_log_probs_2.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)

    # advantage(행동가치(할인 총 보상, discounted reward)-상태가치(critic value)) 계산
    advantages = rollouts.returns[:-1] - values
    advantages -= torch.mean(advantages)
    # Critic의 loss 계산
    value_loss = advantages.pow(2).mean()

    # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
    action_gain_1 = ((action_log_probs_1) * advantages.detach()).mean()
    action_gain_2 = ((action_log_probs_2) * advantages.detach()).mean()
    # detach 메서드를 호출하여 advantages를 상수로 취급

    # 오차함수의 총합
    loss_1 = - action_gain_1  # + (value_loss * VALUE_COEFF - entropy1 * ENTROPY_COEFF)
    loss_2 = - action_gain_2  # + (value_loss * VALUE_COEFF - entropy2 * ENTROPY_COEFF)
    total_loss = (loss_1 + loss_2) / a_size

    model.train()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()  # 결합 가중치 수정
    rollouts.after_update()

    i += 1

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(rlist, 'b')
plt.show()
