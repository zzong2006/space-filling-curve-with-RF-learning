import torch
import torch.nn as nn
import sys
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from LSTM_Sample import init_weights

'''
    1 ~ NUM 까지의 숫자를 중복없이 모두 선택하는 RL Model
    각 숫자는 one-hot-encoding 으로 변환 (즉, input size 는 NUM)
'''


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(Net, self).__init__()
        self.a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.b = nn.Linear(hidden_size, output_size)
        # (state, action, reward, hidden & cells)
        self.ep_history = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.GAMMA = 0.999
        self.optimizer = optim.RMSprop(lr=learning_rate, params=self.parameters())

    def forward(self, input, his=None):
        output, (hidden, cell) = self.a(input, his)
        output = hidden
        # output = torch.relu(output)
        output = self.b(output)

        return output, (hidden, cell)

    def update(self):
        # 정답 신호로 사용할 Q(s_t, a_t)를 계산
        loss = 0
        R = 0
        episode_length = len(self.rewards)
        # ep_history = np.array(self.ep_history)
        for i in reversed(range(episode_length)):
            R = self.GAMMA * R + self.rewards[i]
            reward = torch.FloatTensor([R])
            policy_loss = reward * self.log_probs[i] - 0.01 * self.entropies[i]
            loss = loss - policy_loss
        print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def discount_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.GAMMA + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def episode_reset(self):
        self.ep_history = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self


if __name__ == "__main__":
    NUM = 8
    net = Net(input_size=NUM, hidden_size=100, output_size=NUM, learning_rate=1e-3)
    net.apply(init_weights)
    epochs = 10000
    one_hot = np.eye(NUM)
    done_list = deque(maxlen=10)
    reward_list = deque(maxlen=10)

    for i in range(epochs):
        # 초기 시작 값은 0 [batch = 1, seq = n, input_size = NUM]
        curr_state = Variable(torch.zeros(1, 1, NUM))
        order = []
        curr_his = (Variable(torch.zeros(1, 1, 100)), Variable(torch.zeros(1, 1, 100)))
        done = False

        for j in range(NUM):
            # Action
            logit, his = net(curr_state, curr_his)
            prob = torch.softmax(logit, dim=-1)
            # print(logit.data, prob.data)
            log_prob = torch.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum()
            a = torch.multinomial(prob.view(-1), num_samples=1).data
            log_prob = log_prob.view(-1).gather(0, Variable(a))
            net.entropies.append(entropy)
            net.log_probs.append(log_prob)

            # Reward & Done
            if a.item() in order:
                done = True
                reward = -(1 - (len(order) / NUM) )
            else:
                if j == NUM - 1:
                    reward = 1
                else:
                    reward = 0
                done = False
                order.append(a.item())
            # Make state from order
            state = one_hot[order]
            net.rewards.append(reward)
            # net.ep_history.append([curr_state, np.array([a]), np.array([reward]), curr_his])
            curr_state = torch.FloatTensor(state).view(1, -1, NUM)
            curr_his = his
            if done:
                break
        done_list.append(done)
        if not done:
            print(f'{i} , Result : {done}, state : {order}, reward : {reward}')
        if i > 10:
            print(f'[{i}] 최근 10번 시도 시 성공 비율 : {np.mean(np.logical_not(done_list)) * 100}%')
        net.update()
        net.episode_reset()

    print('TEST')
    sys.exit(0)
