import torch
import torch.nn as nn
import numpy as np
from torch import optim
import math
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

gamma = 0.99


class Environment():
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
            self.a += 0.01
        else:
            self.a -= 0.01

        if action[1] == 0:
            self.b += 0.01
        else:
            self.b -= 0.01

        curr = np.abs(self.a - self.b)
        if math.isclose(self.prev, curr, rel_tol=1e-5):
            reward = 0
        elif self.prev > curr:
            done = True
            # print(self.a, self.b, self.prev, curr)
            reward = -1
        else:
            reward = 1

        self.prev = curr
        return np.array([self.a, self.b, self.prev]), reward, done


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # Normalize reward to avoid a big variability in rewards
    mean = np.mean(discounted_r)
    std = np.std(discounted_r)
    if std == 0:
        std = 1
    normalized_discounted_r = (discounted_r - mean) / std
    return normalized_discounted_r


class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Agent, self).__init__()
        self.a = nn.Linear(input_size, hidden_size, bias=False)
        self.b = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        output = torch.relu(self.a(input))
        output = torch.softmax(self.b(output), dim=-1)
        return output


class Modified_Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Modified_Agent, self).__init__()
        self.a = nn.Linear(input_size, hidden_size)
        self.a_1 = nn.Linear(hidden_size, hidden_size)
        self.a_2 = nn.Linear(hidden_size, output_size)
        self.b = nn.Linear(hidden_size, hidden_size)
        self.b_1 = nn.Linear(hidden_size, hidden_size)
        self.b_2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output_a = torch.relu(self.a(input))
        output = self.a_1(output_a)
        output = self.a_2(output)
        first = torch.softmax(output, dim=-1)
        output = torch.relu(self.b(torch.relu(output_a)))
        output = self.b_1(output)
        second = torch.softmax(self.b_2(output), dim=-1)
        return first, second


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Set total number of episodes to train agent on.
total_episodes = 1000
max_ep = 999
update_frequency = 5

i = 0
total_reward = []
total_length = []
rlist = []
lr = 1e-3
s_size = 3  # state
a_size = 2  # action
h_size = 16  # hidden

model = Modified_Agent(s_size, h_size, a_size)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
env = Environment()

while i < total_episodes:
    if i % 100 == 0:
        print(f'Start {i}th Episode ... ')
    s = env.reset()
    running_reward = 0
    ep_history = []

    for j in range(max_ep):
        # 네트워크 출력에서 확률적으로 액션을 선택
        with torch.no_grad():
            s = torch.from_numpy(s).type(torch.FloatTensor)
            chosen_action_1, chosen_action_2 = model(s)
            a_dist = chosen_action_1.detach().numpy()
            a = np.random.choice(a_dist, p=a_dist)
            a = np.argmax(a_dist == a)
            b_dist = chosen_action_2.detach().numpy()
            b = np.random.choice(b_dist, p=b_dist)
            b = np.argmax(b_dist == b)

        s1, r, d = env.modified_step([a, b])  # Get our reward for taking an action
        ep_history.append([s.numpy(), np.array([a, b]), r, s1])
        s = s1  # Next state
        running_reward += r
        rlist.append(running_reward)

        if d == True:
            # Update Network
            running_reward = 0
            ep_history = np.array(ep_history)
            # 보상을 증식시켜 최근에 얻은 보상이 더 커지게 설정함
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])
            # feed_dict={myAgent.reward_holder:ep_history[:,2],
            # myAgent.action_holder:ep_history[:,1],  myAgent.state_in:np.vstack(ep_history[:,0])}
            state_in = np.vstack(ep_history[:, 0])
            state_in = torch.from_numpy(state_in).type(torch.FloatTensor)

            model.eval()
            output_1, output_2 = model(state_in)
            indexes = np.vstack(ep_history[:, 1])
            indexes_1 = torch.from_numpy(indexes[:, 0].astype('int32')).type(torch.LongTensor)
            indexes_2 = torch.from_numpy(indexes[:, 1].astype('int32')).type(torch.LongTensor)
            reward = torch.from_numpy(ep_history[:, 2].astype('float32')).type(torch.FloatTensor)
            responsible_outputs_1 = output_1.gather(1, indexes_1.view(-1, 1))
            responsible_outputs_2 = output_2.gather(1, indexes_2.view(-1, 1))

            # print(loss)
            model.train()
            loss_1 = -torch.sum(torch.log(responsible_outputs_1.view(-1)) * reward)
            loss_2 = -torch.sum(torch.log(responsible_outputs_2.view(-1)) * reward)
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward.append(running_reward)
            total_length.append(j)
            break
    i += 1

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(rlist, 'b')
plt.show()
