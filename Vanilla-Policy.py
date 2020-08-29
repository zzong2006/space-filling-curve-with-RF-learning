import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

try:
    xrange = xrange
except:
    xrange = range

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Agent, self).__init__()
        self.a = nn.Linear(input_size, hidden_size, bias=False)
        self.b = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        output = torch.relu(self.a(input))
        output = torch.softmax(self.b(output), dim = -1)
        return output

#Set total number of episodes to train agent on.
total_episodes = 5000
max_ep = 999
update_frequency = 5

i = 0
total_reward = []
total_length = []

lr = 1e-2
s_size = 4  # state
a_size = 2  # action
h_size = 8  # hidden

model = Agent(s_size, h_size, a_size)
optimizer = optim.Adam(model.parameters(), lr = lr)


while i < total_episodes:
    s = env.reset()
    running_reward = 0
    ep_history = []

    for j in range(max_ep):
        # 네트워크 출력에서 확률적으로 액션을 선택
        with torch.no_grad():
            s = torch.from_numpy(s).type(torch.FloatTensor)
            chosen_action = model(s)
            a_dist = chosen_action.detach().numpy()
            a = np.random.choice(a_dist, p = a_dist)
            a = np.argmax(a_dist == a)

        s1, r, d, _ = env.step(a) # Get our reward for taking an action
        ep_history.append([s.numpy(), a, r, s1])
        s = s1 # Next state
        running_reward += r

        if d == True:
            # Update Network
            ep_history = np.array(ep_history)
            # 보상을 증식시켜 최근에 얻은 보상이 더 커지게 설정함
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])
            # feed_dict={myAgent.reward_holder:ep_history[:,2],
            # myAgent.action_holder:ep_history[:,1],  myAgent.state_in:np.vstack(ep_history[:,0])}
            state_in = np.vstack(ep_history[:, 0])
            state_in = torch.from_numpy(state_in).type(torch.FloatTensor)

            model.eval()
            output = model(state_in)
            indexes = torch.from_numpy(ep_history[:, 1].astype('Int32')).type(torch.LongTensor)
            reward = torch.from_numpy(ep_history[:, 2].astype('Float32')).type(torch.FloatTensor)
            responsible_outputs = output.gather(1, indexes.view(-1,1))

            # print(loss)
            model.train()
            loss = -torch.mean(torch.log(responsible_outputs.view(-1)) * reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward.append(running_reward)
            total_length.append(j)
            break
    if i % 100 == 0:
        print(np.mean(total_reward[-100:]))
    i += 1
