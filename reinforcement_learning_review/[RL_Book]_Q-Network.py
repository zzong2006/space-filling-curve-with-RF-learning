"""
책에서 소개된 TensorFlow library를 사용한 Q-Learning code를 pyTorch로 변경하여 연습
책 제목: 강화학습 첫 걸음
Github Link: https://github.com/kyoseoksong/RL_Book/blob/master/Chap5-2.Q-Network.ipynb
"""

import torch
import random
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from collections import namedtuple


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = gym.make('FrozenLake-v0')
model = nn.Linear(16, 4)
optimizer = optim.Adam(model.parameters(), lr=0.01)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
memory = ReplayMemory(1000)

# discount factor
y = .999
# epsilon factor
steps_done = 0
e = 0.1
num_episode = 2000
BATCH_SIZE = 128
# 보상의 총계와 에피소드별 단계 수를 담을 리스트를 생성함
jList = []
rList = []

for i in range(num_episode):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # Q Network
    while j < 99:
        j += 1
        # Q Network 에서 (e 의 확률로 랜덤한 액션과 함께) greedy한 액션을 선택
        model.eval()
        with torch.no_grad():
            allQ = model(torch.eye(16)[s:s + 1])  # one-hot-vector 형식으로 현재 상태값 생성
            a = allQ.max(1).indices.item()
            eps_threshold = 0.05 + (0.9 - 0.05) * np.exp(-1. * steps_done / 200)
            steps_done += 1
            if np.random.rand(1) < eps_threshold:
                a = env.action_space.sample()
            # 환경으로부터 새로운 상태와 보상을 얻음
            s1, r, d, _ = env.step(a)
            # 새로운 상태를 네트워크에 feed 해줌으로써 Q' 값을 구함
            Q1 = model(torch.eye(16)[s1:s1 + 1])
            # maxQ' 값을 구하고 선택된 액션에 대한 target 값을 설정
            maxQ1 = Q1.max(1).values
        targetQ = allQ
        targetQ[0, a] = r + y * maxQ1

        # transitions = memory.sample(BATCH_SIZE)

        # target 및 prediction Q 값을 이용해 네트워크를 학습시킴
        model.train()
        Qout = model(torch.eye(16)[s:s + 1])
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(Qout, targetQ)
        loss.backward()
        optimizer.step()
        rAll += r
        s = s1
        if d == True:
            e = 1. / ((i / 50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

print("Percent of successful episodes: " + str(sum(rList) / num_episode))

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(rList, 'b')
ax2.plot(jList, 'r')
plt.show()
