import torch
import torch.nn as nn
import sys
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from collections import deque
from LSTM_Sample import init_weights, save_model, init_weights_from_loaded

'''
    1 ~ NUM 까지의 숫자를 중복없이 모두 선택하는 RL Model
    각 숫자는 one-hot-encoding 으로 변환 (즉, input size 는 NUM)

    rev1 : Reward 와 GAMMA 를 조금 수정함
            * GAMMA 가 너무 높다면, policy 자체는 거의 optimal 인데도, 해당 policy (-) 쪽으로 잘 안가려는 경향이 강함
            * GAMMA 가 낮다면, 문제 특성상 중복이 없이 선택해야 하므로, 특정 action 에 대해 강한 감점을 줄 필요가 있음
            --> 감점은 -1 점으로 수정 (원래는 선택한 숫자의 길이가 큰 만큼 감점을 완화했음)
'''


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, learning_rate):
        super(Net, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.a = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)
        # (state, action, reward, hidden & cells)

        self.GAMMA = 0.99
        self.value_coeff = 1
        self.entropy_coeff = 0.01
        self.DEVICE = 'cpu'
        self.optimizer = optim.Adam(lr=learning_rate, params=self.parameters())

    def forward(self, input, his=None):
        embeds = self.emb(input)
        output, (hidden, cell) = self.a(embeds.view(1, len(input), -1), his)
        # output [ :, -1, : ] 와 hidden 값은 같다.
        output = torch.relu(output[:, -1, :])
        # output = torch.relu(output)
        a_output = self.actor(output)
        c_output = self.critic(output)

        return a_output, c_output, (hidden, cell)

    def update(self):
        # 정답 신호로 사용할 Q(s_t, a_t)를 계산
        loss = 0
        R = 0
        episode_length = len(self.rewards)
        # ep_history = np.array(self.ep_history)
        for i in reversed(range(episode_length)):
            R = self.GAMMA * R + self.rewards[i]
            reward = torch.FloatTensor([R]).to(self.DEVICE)
            policy_loss = reward * self.log_probs[i] - 0.01 * self.entropies[i]
            loss = loss - policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def episode_reset(self):
        self.ep_history = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self

class MemoryBuffer():
    def __init__(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

'''

'''

if __name__ == "__main__":
    NUM = 17
    ACTOR_NUM = 8
    HIDDEN_NUM = 30
    HORIZON = 5
    LR = 25e-4
    net = Net(vocab_size=NUM + 1, embedding_size=10, hidden_size=HIDDEN_NUM, output_size=NUM, learning_rate=LR)
    epochs = 10000
    done_list = deque(maxlen=10)
    reward_list = deque(maxlen=10)

    order = [[] for _ in range(ACTOR_NUM)]
    curr_his = [(Variable(torch.zeros(1, 1, HIDDEN_NUM)), Variable(torch.zeros(1, 1, HIDDEN_NUM))) for _ in range(ACTOR_NUM) ]
    curr_state = [Variable(torch.zeros(1)).long() + NUM for _ in range(ACTOR_NUM)]
    done = [False for _ in range(ACTOR_NUM)]

    for i in range(epochs):
        # 초기 시작 값은 0 [batch = 1, seq = n, input_size = NUM]
        for j in range(ACTOR_NUM):
            for w in range(HORIZON):
                # Action
                logit, value, his = net(curr_state[j], curr_his[j])
                prob = torch.softmax(logit, dim=-1)
                # print(logit.data, prob.data)
                log_prob = torch.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum()
                m = Categorical(prob.view(-1))
                a = m.sample().item()
                log_prob = log_prob.view(-1).gather(0, Variable(a))
                net.entropies.append(entropy)
                net.log_probs.append(log_prob)

                # Reward & Done
                if a.item() in order:
                    done = True
                    reward = -1
                else:
                    if j == NUM - 1:
                        reward = 10
                    else:
                        reward = 1
                    done = False
                    order.append(a.item())

                # Make state from order
                # state = ((( (np.array(order) + 1)[:, None] & (1 << np.arange(NUM)))) > 0).astype('int')
                # state = one_hot[order]
                curr_state = torch.LongTensor(order)
                net.rewards.append(reward)
                # curr_state = torch.FloatTensor(state).view(1, -1, NUM)
                curr_his[j] = his
                if done:
                    break
        done_list.append(done)
        if not done or done:
            print(f'{i} , Result : {done}, state : {np.reshape(order, (-1))}, reward : {reward}')
        if i > 10:
            print(f'[{i}] 최근 10번 시도 시 성공 비율 : {np.mean(np.logical_not(done_list)) * 100}%')
        net.update()
        net.episode_reset()

    save_model(net, './model/lstm.pt')
    print("[After Training] Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        print(torch.mean(net.state_dict()[param_tensor]).item(), torch.std(net.state_dict()[param_tensor]).item())
    print("Optimizer's state_dict:")
    for var_name in net.optimizer.state_dict():
        print(var_name, "\t", net.optimizer.state_dict()[var_name])
    sys.exit(0)
