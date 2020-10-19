import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

'''
 * 08-22 : 기존 방식은 locality가 조금이라도 떨어지면 바로 다음 episode로 넘어갔기 때문에 충분히 학습할 여유가 되지않음
            그렇기 때문에 목숨 개념을 추가해서, 최대 x 번까지 locality가 떨어지게 하는 action을 해도 감점만 하고 지나감
            그리고 Agent가 가진 기회가 끝났는지에 대한 내용도 정보에 추가함 
            ( 현재는 True, False로 하는데 이후 그냥 목숨인 양의 정수 값으로 추가할 것인지는 고려 중 에 있음)

 * 08-26 : - 점수 부여 방식을 조금 수정했는데, 최소 locality가 action을 통해서 나오지 않더라도, 이전의 locality 보다 났다면,
            조금의 점수를 부여하는 방향으로 바꿨음. 다만 이전의 locality 보다 같으면 감점을 부여함
           - 초기 curve를 zig-zag 뿐만 아니라 hilbert 또는 z-curve로 시작하게끔 수정

 * 08-27 : 훈련된 모델을 테스트해볼 수 있도록 test 함수를 Environment 클래스에 추가
 
 * 08-29 : 신경망을 하나로 합쳐서 학습이 잘 되는지 테스트 (잘 안됨!)
 
'''

try:
    xrange = xrange
except:
    xrange = range

NOTEBOOK = False
CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 2
DATA_SIZE = 5
MAX_STEP = 200
CAPACITY = 10000
GAMMA = 0.99  # 시간 할인율
INIT_CURVE = 'hilbert'
LEARNING_RATE = 1e-3  # 학습률
OFFSET = 0  # 기존 state 좌표 값 외에 신경망에 추가로 들어갈 정보의 갯수


class HilbertCurve():
    def __init__(self, dimension):
        self.DIM = dimension

    # convert (x,y) to d
    def xy2d(self, n, x, y):
        d = 0
        s = n // 2
        while s > 0:
            rx = ((x & s) > 0);
            ry = ((y & s) > 0);
            d += s * s * ((3 * rx) ^ ry)
            x, y = self.rot(n, x, y, rx, ry)
            s = s // 2
        return d

    def d2xy(self, n, d):
        t = d
        x = 0
        y = 0
        s = 1
        while s < n:
            rx = 1 & t // 2
            ry = 1 & t ^ rx
            x, y = self.rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t = t // 4
            s *= 2
        return [x, y]

    def rot(self, n, x, y, rx, ry):
        if (ry == 0):
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            t = x
            x = y
            y = t
        return x, y

    def getCoords(self, order):
        N = 2 ** (order * self.DIM)
        coordinates = list(map(self.d2xy, [N] * (N), range(N)))
        return coordinates

'''
생성된 SFC와 비교하기 위한 Z curve
'''

class ZCurve():
    def __init__(self, dimension):
        self.DIM = dimension

    def part1by1(self, n):
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def unpart1by1(self, n):
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0f0f0f0f
        n = (n ^ (n >> 4)) & 0x00ff00ff
        n = (n ^ (n >> 8)) & 0x0000ffff
        return n

    def part1by2(self, n):
        n &= 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n << 8)) & 0x0300f00f
        n = (n ^ (n << 4)) & 0x030c30c3
        n = (n ^ (n << 2)) & 0x09249249
        return n

    def unpart1by2(self, n):
        n &= 0x09249249
        n = (n ^ (n >> 2)) & 0x030c30c3
        n = (n ^ (n >> 4)) & 0x0300f00f
        n = (n ^ (n >> 8)) & 0xff0000ff
        n = (n ^ (n >> 16)) & 0x000003ff
        return n

    # 2 차원 데이터를 비트로 변환하고 교차 생성
    def interleave2(self, x, y):
        return self.part1by1(x) | (self.part1by1(y) << 1)

    # 교차 생성된 값을 2 차원 데이터로 되돌림
    def deinterleave2(self, n):
        return [self.unpart1by1(n), self.unpart1by1(n >> 1)]

    def interleave3(self, x, y, z):
        return self.part1by2(x) | (self.part1by2(y) << 1) | (self.part1by2(z) << 2)

    def deinterleave3(self, n):
        return [self.unpart1by2(n), self.unpart1by2(n >> 1), self.unpart1by2(n >> 2)]

    def getCoords(self, order):
        # temp_index = np.arange(2**(self.DIM * order))
        coords = list(map(self.deinterleave2, np.arange(2 ** (self.DIM * order))))
        return np.array(coords)

'''
초기 SFC 생성 함수 : 이후 class 형태로 바꿀거임
'''


def build_init_coords(order, dimension, init_curve):
    if init_curve == 'zig-zag':
        whole_index = np.arange(2 ** (order * dimension))
        side = np.sqrt(2 ** (order * dimension)).astype(int)
        coords = np.array(list(map(lambda x: list([x // (side), x % (side)]), whole_index)))
    elif init_curve == 'hilbert':
        coords = HilbertCurve(dimension=dimension).getCoords(order=order)
    elif init_curve == 'z':
        coords = ZCurve(dimension=dimension).getCoords(order=order)
    return coords


'''
Grid (회색 선) 을 그릴 좌표를 써주는 함수
Arg : pmax 값
'''


def getGridCooridnates(num):
    grid_ticks = np.array([0, 2 ** num])
    for _ in range(num):
        temp = np.array([])
        for i, k in zip(grid_ticks[0::1], grid_ticks[1::1]):
            if i == 0:
                temp = np.append(temp, i)
            temp = np.append(temp, (i + k) / 2)
            temp = np.append(temp, k)
        grid_ticks = temp
    grid_ticks -= 0.5
    return grid_ticks


def showPoints(data, ax=None, index=True):
    ax = ax or plt.gca()
    pmax = np.ceil(np.log2(np.max(data)))
    pmax = pmax.astype(int)
    offset = 0.5
    cmin = 0
    cmax = 2 ** (pmax) - 1
    side = np.sqrt(2 ** (ORDER * DIM)).astype(int)

    grid_ticks = getGridCooridnates(pmax)

    ax.set_yticks(grid_ticks, minor=False)
    ax.set_xticks(grid_ticks, minor=False)
    plt.xlim(cmin - offset, cmax + offset)
    plt.ylim(cmin - offset, cmax + offset)
    ax.grid(alpha=0.5)

    if index:
        coordinates = np.array(list(map(lambda x: list([x // (side), x % (side)]), data)))
    else:
        coordinates = data

    ax.plot(coordinates[:, 0], coordinates[:, 1], 'o')
    print(f'pmax: {pmax}')


def showlineByIndexorder(data, ax=None, index=True):
    ax = ax or plt.gca()
    side = np.sqrt(2 ** (ORDER * DIM))
    if index:
        coordinates = np.array(list(map(lambda x: list([x // (side), x % (side)]), data)))
    else:
        coordinates = data

    ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=1, linestyle='--')


def changeIndexOrder(indexD, a, b):
    a = a.cpu().numpy().astype(int).item()
    b = b.cpu().numpy().astype(int).item()

    indexD[[a, b]] = indexD[[b, a]]
    return indexD


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    # Normalize reward to avoid a big variability in rewards
    mean = np.mean(discounted_r)
    std = np.std(discounted_r)
    if std == 0: std = 1
    normalized_discounted_r = (discounted_r - mean) / std
    return normalized_discounted_r


'''
SFC를 만드는 모델

Notice
1. dropout은 쓸지 말지 고민중임
2. embedding vector를 사용할지 말지 고민하고 있음 (각 데이터의 좌표로 유사성을 파악하기)
'''


class SFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SFCNet, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.first = nn.Linear(input_size, self.hidden_size)
        self.first_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.first_out = nn.Linear(self.hidden_size, output_size)


    def forward(self, input):
        output = torch.relu(self.first(input))
        output = torch.relu(self.first_add(output))
        first = self.first_out(output)
        first_output = torch.softmax(first, dim=-1)
        return first_output


class Brain:
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions

        self.model = SFCNet(num_states, hidden_size, num_actions)
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

        output = self.model(state_in)
        indexes = np.vstack(ep_history[:, 1])
        indexes = torch.from_numpy(indexes.astype('int32')).type(torch.cuda.LongTensor)

        reward = torch.from_numpy(ep_history[:, 2].astype('float32')).type(torch.cuda.FloatTensor)
        responsible_outputs = output.gather(1, indexes.view(-1, 2))

        # print(loss)
        self.model.train()
        loss = -torch.mean(responsible_outputs.log() * reward.view(-1,1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        with torch.no_grad():
            output = self.model(state.view(1, -1))
            choices = output.multinomial(2)

            return choices.float()


class Agent():
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_policy_function(self, history):
        self.brain.update(history)

    def get_action(self, state, step):
        action = self.brain.decide_action(state, step)
        return action


class Env():
    def __init__(self, data_index, order, max_episode, max_step, init_curve, dimension=2):
        self.DIM = dimension
        self.iteration = order
        self.MAX_STEP = max_step
        self.MAX_EPISODE = max_episode
        self.data_index = data_index
        # self.initial_curve = init_curve

        self.num_action_space = 2 ** (dimension * order)
        self.num_observation_space = 2 ** (dimension * order) * 3 + OFFSET

        # Reward 설정용
        self.init_coords = build_init_coords(order, dimension, init_curve)
        self.agent = Agent(self.num_observation_space, self.num_action_space)
        self.analyzer = Analyzer(data_index, self.init_coords.copy(), order=order, dim=dimension)
        self.hilbert = HilbertCurve(dimension=dimension)
        self.z = ZCurve(dimension=dimension)

    '''
    초기 state를 생성하는 함수; 
    1. 활성화된 데이터의 binary 표현
    2. query area 단위 따른 clustering 갯수 (max, average) : (미구현) 
    3. 전체 area에서 curve를 따라서 모든 활성화된 데이터를 지날 수 있는 curve의 최소 길이 (또는 query area에서만 구할 수 있는 길이) : (미구현) 
    '''

    def reset(self, data_index):
        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[data_index] = 1
        observation = np.concatenate((avail, self.init_coords), axis=1)
        return observation

    '''
    Agent로 부터 행동을 선택하고 그 행동에 맞춰서 이후 상태 관찰,
    관찰에 기반하여 보상을 계산하고 이들을 버퍼에 저장
    버퍼에 충분히 Transition을 저장했다고 생각하면 신경망(Q 함수) 업데이트
    '''

    def run(self):
        span = 100
        locality_average_list = []
        locality_per_episode_list = []
        locality_per_step_list = []
        episode_list = np.zeros(span)
        reward_list = np.zeros(span)

        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[self.data_index] = 1

        h_state = np.concatenate((avail, self.hilbert.getCoords(self.iteration)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(self.iteration)), axis=1)
        h_num = self.analyzer.l2NormLocality(h_state)
        z_num = self.analyzer.l2NormLocality(z_state)

        self.observation = self.reset(self.data_index)
        min_o_num = self.analyzer.l2NormLocality(self.observation)
        global_min_o_num = min_o_num  # 최소 locality 의 역(reverse) 값
        global_min_state = np.empty(1)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial : {min_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            self.observation = self.reset(self.data_index)
            min_o_num = self.analyzer.l2NormLocality(self.observation)
            prev_o_num = -1;
            prev_action = torch.cuda.FloatTensor([[-1, -1]])
            # 추가 정보 부여 (후에 함수로 만들 것)
            # observation = np.append(self.observation.reshape(1, -1), min_o_num)
            # observation = np.append(observation, 0)
            observation = self.observation
            state = torch.from_numpy(observation).type(torch.cuda.FloatTensor)
            state = state.view(1, -1)
            total_sum = 0
            total_reward = 0
            ep_history = []
            done = False
            life = 5

            for step in range(self.MAX_STEP):
                action = self.agent.get_action(state, episode)
                self.observation_next = self.step(action)
                o_num = self.analyzer.l2NormLocality(self.observation_next)
                locality_per_step_list.append(o_num)
                total_sum += o_num

                # Update Minimum reverse of the locality
                if global_min_o_num > o_num:
                    global_min_o_num = o_num
                    global_min_state = self.observation_next.copy()

                # Reward Part
                # print(f'({action[0][0].data},{action[0][1].data})')
                if action[0][0].data == action[0][1].data or \
                        ((prev_action[0][0].data == action[0][0].data) and (
                                prev_action[0][1].data == action[0][1].data)):
                    reward = torch.cuda.FloatTensor([-10])
                else:
                    if min_o_num < o_num:
                        if prev_o_num == -1 or prev_o_num <= o_num:
                            life -= 1
                            reward = torch.cuda.FloatTensor([-0.5])
                            if life <= 0:
                                done = True
                        elif prev_o_num > o_num:
                            reward = torch.cuda.FloatTensor([0.0])
                    elif min_o_num == o_num:
                        reward = torch.cuda.FloatTensor([-0.05])
                    else:
                        reward = torch.cuda.FloatTensor([min_o_num / o_num * 10])
                        min_o_num = o_num
                prev_o_num = o_num
                prev_action = action
                # state에 점 사이 locality 정보와 목숨 내용을 추가함
                # observation_next = np.append(self.observation_next.reshape(1, -1), o_num)
                # observation_next = np.append(observation_next, 0 if done == False else 1)
                observation_next = self.observation_next
                total_reward += (reward.detach().cpu().numpy())
                ep_history.append([state.detach().cpu().numpy(), action.detach().cpu().numpy(),
                                   reward.detach().cpu().numpy(), observation_next])
                state_next = observation_next
                state_next = torch.from_numpy(state_next).type(torch.cuda.FloatTensor)
                state_next = state_next.view(1, -1)
                state = state_next

                if done:
                    self.agent.update_policy_function(ep_history)
                    break

            episode_list = np.hstack((episode_list[1:], total_sum / (step + 1)))
            reward_list = np.hstack((reward_list[1:], total_reward))
            locality_per_episode_list.append(total_sum / (step + 1))
            if episode > span:
                locality_average_list.append(episode_list.mean())
                print(f'episode {episode} is over within step {step + 1}. \n'
                      f'Average of the cost in the {span} episodes : {episode_list.mean()} And Reward : {reward_list.mean()}')
            if episode % 1000 == 0:
                print(f'{episode}, Recorded the minimum reverse of the locality so far: {global_min_o_num} ')
        if NOTEBOOK:
            f = plt.figure(figsize=(30, 8))

            ax1 = f.add_subplot(311)
            plt.plot(locality_per_episode_list, 'r-')
            plt.xlabel('episode')
            plt.ylabel('Reverse of the locality')

            ax2 = f.add_subplot(312)
            plt.plot(locality_per_step_list, 'b-')
            plt.xlabel('step')
            plt.ylabel('Reverse of the locality')

            ax2 = f.add_subplot(313)
            plt.plot(locality_average_list, 'r-')
            plt.xlabel('episode')
            plt.ylabel(f'(Average per {span})Reverse of the locality')

            plt.tight_layout()
            plt.show(block=True)

        return global_min_o_num, global_min_state

    def test(self, data_index, max_episode, max_step):
        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[data_index] = 1

        TestAnalyzer = Analyzer(data_index, self.init_coords.copy(), order=self.iteration, dim=self.DIM)

        h_state = np.concatenate((avail, self.hilbert.getCoords(self.iteration)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(self.iteration)), axis=1)
        h_num = TestAnalyzer.l2NormLocality(h_state)
        z_num = TestAnalyzer.l2NormLocality(z_state)

        self.observation = self.reset(data_index)
        min_o_num = TestAnalyzer.l2NormLocality(self.observation)
        global_min_o_num = min_o_num  # 최소 locality 의 역(reverse) 값
        global_min_state = np.empty(1)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial : {min_o_num}')

        for episode in range(max_episode):
            self.observation = self.reset(data_index)
            min_o_num = TestAnalyzer.l2NormLocality(self.observation)
            observation = self.observation
            state = torch.from_numpy(observation).type(torch.cuda.FloatTensor)
            state = state.view(1, -1)
            done = False
            life = 1
            for step in range(max_step):
                action = self.agent.get_action(state, episode)
                self.observation_next = self.step(action)
                o_num = TestAnalyzer.l2NormLocality(self.observation_next)

                # Update Minimum reverse of the locality
                if global_min_o_num > o_num:
                    global_min_o_num = o_num
                    global_min_state = self.observation_next.copy()

                state_next = self.observation_next
                state_next = torch.from_numpy(state_next).type(torch.cuda.FloatTensor)
                state_next = state_next.view(1, -1)
                state = state_next

                if done:
                    break
            print(f'episode {episode} is over within step {step + 1}. \n'
                  f'Recorded the minimum reverse of the locality so far: {global_min_o_num}')

        return global_min_o_num, global_min_state

    '''
    주어진 action 을 수행하고 난 뒤의 state를 반환
    '''

    def step(self, choosenAction):
        next_state = changeIndexOrder(self.observation, choosenAction[:, 0], choosenAction[:, 1])
        return next_state


'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


class Analyzer():
    def __init__(self, index, init_state, order, dim):
        self.iteration = order
        self.DIM = dim
        self.scan_index = index

        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[index] = 1

        self.init_state = np.concatenate((avail, init_state), axis=1)

    '''
    전체 활성화 데이터를 모두 거치는데 필요한 path의 최소 비용을 계산함
    '''

    def miniPath(self, compared_state):
        # 활성화된 데이터의 좌표값을 기준으로 현재 변경된 좌표값을 변화를 줌 (의미가 있는지?)
        self.sort(self.init_state, compared_state)

        onlyIndex = compared_state[:, 0]
        reverseIndex = onlyIndex[::-1]
        start_idx = np.argmax(onlyIndex == 1)
        end_idx = len(reverseIndex) - np.argmax(reverseIndex == 1) - 1
        cost = end_idx - start_idx
        return cost

    '''
    각 활성화 데이터간 존재하는 거리(비용)을 모두 더한 값을 반환
    '''

    def sumEachPath(self, compared_state):
        self.sort(self.init_state, compared_state)

        onlyIndex = compared_state[:, 0]
        prev = -1
        cost = 0
        for i, v in enumerate(onlyIndex):
            if v == 1:
                if prev != -1:
                    cost += (i - prev)
                prev = i
        return cost

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
        moved_argsorted = np.lexsort((moved[:, 1], moved[:, 2]))
        init_argsorted = np.lexsort((init[:, 1], init[:, 2]))
        moved[moved_argsorted, 0] = init[init_argsorted, 0]
        return moved


'''
index (n) 은 다음과 같이 좌표로 표시됨
n 의 최댓값은 DIM * ORDER - 1 
좌표 값은 ( n // (DIM * ORDER), n % (DIM * ORDER) ) 
'''


def main():
    np.random.seed(210)

    side = np.sqrt(2 ** (ORDER * DIM))
    scan_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
    sample_data = np.array(list(map(lambda x: list([x // (side), x % (side)]), scan_index)))
    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)

        if INIT_CURVE == 'hilbert':
            showlineByIndexorder(np.array(HilbertCurve(DIM).getCoords(ORDER)), ax, index=False)
        elif INIT_CURVE == 'zig-zag':
            grid_index = np.arange(2 ** (ORDER * DIM))
            showlineByIndexorder(grid_index, ax)
        plt.show(block=True)

    env = Env(data_index=scan_index, order=ORDER, max_episode=2000, max_step=10,
              init_curve=INIT_CURVE, dimension=DIM)
    result_value, result_state = env.run()

    print(f'Recorded the minimum reverse of the locality :{result_value}')
    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)
        showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
        plt.show(block=True)

    # Test trained model
    np.random.seed(175)

    print(f'Start testing trained model ... ')
    test_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
    result_value, result_state = env.test(test_index, max_episode=1000, max_step=100)

    print(f'[TEST]Recorded the minimum reverse of the locality :{result_value}')
    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)
        showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
        plt.show(block=True)


if __name__ == '__main__':
    main()