import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from curve import HilbertCurve, ZCurve
from utils import *
from rl_network import PolicyGradient
from environment import CurveEnvironment

'''
 * 08-26 : state 를 normalize 한 버전. Locality 계산을 위해 original 버전은 지울 수 없고, 
            그냥 normalized 한 버전과 계산 및  illustrate 용은 따로 두기로 한다.
            또한 state 에 들어있는 locality 값과 life 존재 여부를 지웠다.
            왜냐하면 실험 결과 해당 값들이 들어 있는지의 여부는 결과에 큰 영향을 미치지 않기 때문임
        => 실험 결과 state를 normalize 한 것은 크게 의미가 있는 것 같지 않음. 오히려 학습 속도를 떨어트림
            대신 학습 경향이 조금 바뀌었는데, 학습이 안되다가 갑자기 되는 불안정한 현상을 보임
            이전에는 안정적으로 가다가 어느 지점에 다다르면 학습이 안되는 경우를 보였음 (original file 참조)
'''

CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 3
DATA_SIZE = 15
MAX_STEP = 200
GAMMA = 0.99  # 시간 할인율
LEARNING_RATE = 1e-4  # 학습률

'''
초기 SFC 생성 함수 : 이후 class 형태로 바꿀거임
'''


def build_init_state(order, dimension):
    whole_index = np.arange(2 ** (order * dimension))
    side = np.sqrt(2 ** (order * dimension)).astype(int)
    coords = np.array(list(map(lambda x: list([x // (side), x % (side)]), whole_index)))
    return coords


'''
Grid (회색 선) 을 그릴 좌표를 써주는 함수
Arg : pmax 값
'''




def changeIndexOrder(indexD, a, b):
    a = a.cpu().numpy().astype(int).item()
    b = b.cpu().numpy().astype(int).item()

    indexD[[a, b]] = indexD[[b, a]]
    return indexD


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


def normalize_state(state):
    state[:, 1] = state[:, 1] / np.linalg.norm(state[:, 1])
    state[:, 2] = state[:, 2] / np.linalg.norm(state[:, 2])
    return state


'''
SFC를 만드는 모델

Notice
1. dropout은 쓸지 말지 고민중임
2. embedding vector를 사용할지 말지 고민하고 있음 (각 데이터의 좌표로 유사성을 파악하기)
'''


class Brain:
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions

        self.model = PolicyGradient(num_states, hidden_size, num_actions)
        if CUDA:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        print(self.model)

    '''
    Policy Gradient 알고리즘으로 신경망의 결합 가중치 학습
    '''

    def update(self, ep_history):
        # 정답 신호로 사용할 Q(s_t, a_t)를 계산
        self.model.eval()
        ep_history = np.array(ep_history)

        ep_history[:, 2] = discount_rewards(ep_history[:, 2])
        state_in = np.vstack(ep_history[:, 0])
        state_in = torch.from_numpy(state_in).type(torch.cuda.FloatTensor)

        output_1, output_2 = self.model(state_in)
        indexes = np.vstack(ep_history[:, 1])
        indexes_1 = torch.from_numpy(indexes[:, 0].astype('Int32')).type(torch.cuda.LongTensor)
        indexes_2 = torch.from_numpy(indexes[:, 1].astype('Int32')).type(torch.cuda.LongTensor)
        reward = torch.from_numpy(ep_history[:, 2].astype('Float32')).type(torch.cuda.FloatTensor)
        responsible_outputs_1 = output_1.gather(1, indexes_1.view(-1, 1))
        responsible_outputs_2 = output_2.gather(1, indexes_2.view(-1, 1))

        self.model.train()
        loss_1 = -torch.sum(torch.log(responsible_outputs_1.view(-1)) * reward)
        loss_2 = -torch.sum(torch.log(responsible_outputs_2.view(-1)) * reward)
        loss = loss_1 + loss_2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        with torch.no_grad():
            first, second = self.model(state.view(1, -1))
            a_dist = first.cpu().detach().numpy().reshape([-1])
            a = np.random.choice(a_dist, p=a_dist)
            a = np.argmax(a_dist == a)
            b_dist = second.cpu().detach().numpy().reshape([-1])
            b = np.random.choice(b_dist, p=b_dist)
            b = np.argmax(b_dist == b)

            return torch.cuda.FloatTensor([[a, b]])


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_policy_function(self, history):
        self.brain.update(history)

    def get_action(self, state, step):
        action = self.brain.decide_action(state, step)
        return action


class Env:
    def __init__(self, data_index, order, max_episode, max_step, dimension=2):
        self.DIM = dimension
        self.iteration = order
        self.MAX_STEP = max_step
        self.MAX_EPISODE = max_episode
        self.data_index = data_index

        self.num_action_space = 2 ** (dimension * order)
        self.num_observation_space = 2 ** (dimension * order) * 3 + OFFSET

        # Reward 설정용
        init_state = build_init_state(order, dimension)
        self.agent = Agent(self.num_observation_space, self.num_action_space)
        self.analyzer = Analyzer(data_index, init_state, order=order, dim=dimension)
        self.hilbert = HilbertCurve(dimension=dimension)
        self.z = ZCurve(dimension=dimension)

    '''
    초기 state를 생성하는 함수; 
    1. 활성화된 데이터의 binary 표현
    2. query area 단위 따른 clustering 갯수 (max, average) : (미구현) 
    3. 전체 area에서 curve를 따라서 모든 활성화된 데이터를 지날 수 있는 curve의 최소 길이 (또는 query area에서만 구할 수 있는 길이) : (미구현) 
    '''

    def reset(self):
        whole_index = np.arange(2 ** (self.iteration * self.DIM))
        side = np.sqrt(2 ** (self.iteration * self.DIM)).astype(int)
        self.whole_index = np.array(list(map(lambda x: list([x // (side), x % (side)]), whole_index)))
        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[self.data_index] = 1
        observation = np.concatenate((avail, self.whole_index), axis=1)
        return observation

    '''
    신경망 (Policy Gradient) 업데이트
    '''

    def run(self):
        span = 10
        locality_list = []
        episode_list = np.zeros(span)
        reward_list = np.zeros(span)

        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[self.data_index] = 1

        h_state = np.concatenate((avail, self.hilbert.getCoords(self.iteration)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(self.iteration)), axis=1)
        h_num = self.analyzer.l2NormLocality(h_state)
        z_num = self.analyzer.l2NormLocality(z_state)

        self.observation = self.reset()
        min_o_num = self.analyzer.l2NormLocality(self.observation)
        global_min_o_num = min_o_num  # 최소 locality 의 역(reverse) 값
        global_min_state = np.empty(1)
        print(f'hilbert : {h_num} , Z : {z_num}, Initial : {min_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            self.observation = self.reset()
            original_state = self.observation.copy()  # Normalize 하기 전의 state 값
            min_o_num = self.analyzer.l2NormLocality(original_state)
            prev_o_num = -1

            observation = normalize_state(self.observation)
            state = torch.from_numpy(observation).type(torch.cuda.FloatTensor)
            state = state.view(1, -1)
            total_sum = 0
            total_reward = 0
            ep_history = []
            done = False
            life = 5

            for step in range(self.MAX_STEP):
                action = self.agent.get_action(state, episode)
                self.observation_next = self.step(self.observation, action)
                original_state = self.step(original_state, action)

                o_num = self.analyzer.l2NormLocality(original_state)
                total_sum += o_num

                # Update Minimum reverse of the locality
                if global_min_o_num > o_num:
                    global_min_o_num = o_num
                    global_min_state = original_state.copy()

                # Reward Part
                if min_o_num < o_num:
                    if prev_o_num == -1 or prev_o_num <= o_num:
                        life -= 1
                        reward = torch.cuda.FloatTensor([-10])
                        if life <= 0:
                            done = True
                    elif prev_o_num > o_num:
                        reward = torch.cuda.FloatTensor([10])
                elif min_o_num == o_num:
                    reward = torch.cuda.FloatTensor([0.0])
                else:
                    reward = torch.cuda.FloatTensor([100])
                    min_o_num = o_num

                prev_o_num = o_num
                # state에 점 사이 locality 정보와 목숨 내용을 추가함
                # observation_next = np.append(self.observation_next.reshape(1, -1), o_num)
                # observation_next = np.append(observation_next, 0 if done == False else 1)
                total_reward += (reward.detach().cpu().numpy())
                ep_history.append([state.detach().cpu().numpy(), action.detach().cpu().numpy(),
                                   reward.detach().cpu().numpy(), np.array([[], []])])
                state_next = self.observation_next.copy()
                state_next = torch.from_numpy(state_next).type(torch.cuda.FloatTensor)
                state_next = state_next.view(1, -1)
                state = state_next

                if done:
                    self.agent.update_policy_function(ep_history)
                    break

            episode_list = np.hstack((episode_list[1:], total_sum / (step + 1)))
            reward_list = np.hstack((reward_list[1:], total_reward))
            if episode > span:
                locality_list.append(episode_list.mean())
                print(f'episode {episode} is over within step {step + 1}. '
                      f'Average of the cost in the {span} episodes : {episode_list.mean()} And Reward : {reward_list.mean()}')

        return global_min_o_num, global_min_state

    '''
    주어진 action 을 수행하고 난 뒤의 state를 반환
    '''

    def step(self, state, choosenAction):
        next_state = changeIndexOrder(state, choosenAction[:, 0], choosenAction[:, 1])
        return next_state


'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


class Analyzer:
    def __init__(self, index, init_state, order, dim):
        self.iteration = order
        self.DIM = dim
        self.scan_index = index

        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[index] = 1

        self.init_state = np.concatenate((avail, init_state), axis=1)

    def l2NormLocality(self, compared_state):
        self.sort(self.init_state, compared_state)

        # 활성화된 데이터만 모음, 결과는 (x, y, 데이터 순서)
        avail_data = np.array([np.append(x[1:], np.array([i])) for i, x in enumerate(compared_state) if x[0] == 1])
        cost = 0

        for (x, y) in combinations(avail_data, 2):
            dist_2d = np.sum((x[0:2] - y[0:2]) ** 2)
            dist_1d = np.abs(x[2] - y[2])
            # Locality Ratio 가 1과 가까운지 측정
            cost += np.abs(1 - (dist_1d / dist_2d))

        return cost

    def sort(self, init, moved):
        # set_trace()
        moved_argsorted = np.lexsort((moved[:, 1], moved[:, 2]))
        init_argsorted = np.lexsort((init[:, 1], init[:, 2]))
        moved[moved_argsorted, 0] = init[init_argsorted, 0]
        return moved


if __name__ == '__main__':
    '''
    index (n) 은 다음과 같이 좌표로 표시됨
    n 의 최댓값은 DIM * ORDER - 1 
    좌표 값은 ( n // (DIM * ORDER), n % (DIM * ORDER) ) 
    '''

    np.random.seed(210)

    side = np.sqrt(2 ** (ORDER * DIM))
    scan_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
    sample_data = np.array(list(map(lambda x: list([x // (side), x % (side)]), scan_index)))
    # print(sample_index,'\n', sample_data)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    # show_points(sample_data, ax, index=False)
    grid_index = np.arange(2 ** (ORDER * DIM))
    # show_line_by_index_order(grid_index, ax)
    # plt.show(block=True)

    env = Env(data_index=scan_index, order=ORDER, max_episode=5000, max_step=1000, dimension=DIM)
    result_value, result_state = env.run()

    print(f'Recorded the minimum reverse of the locality :{result_value}')
    # fig, ax = plt.subplots(1, figsize=(10,10))
    # showPoints(sample_data, ax, index=False)
    # showlineByIndexorder(result_state[:,1:3].reshape([-1,2]), ax ,index = False)
    #
    # plt.show(block=True)
