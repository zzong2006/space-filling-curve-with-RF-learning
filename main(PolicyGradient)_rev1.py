import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

'''
Rev1 에서 수정된 점
 * 기존 파일 main(PolicyGradient) 에서 가능한 action 을 특정 cell 선택 후 상하좌우에서만 바꿀 수 있도록 변경
구체적 수정 사항
 * RNN 에서 두번째 신경망 출력의 크기를 4로 변경 (상하좌우)
 * Environment 의 step 함수를 수정
'''

try:
    xrange = xrange
except:
    xrange = range

CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 3
DATA_SIZE = 5
OFFSET = 1
MAX_STEP = 200
CAPACITY = 10000
GAMMA = 0.99  # 시간 할인율
LEARNING_RATE = 1e-3 # 학습률

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
def build_init_state(order, dimension):
    whole_index = np.arange(2**(order * dimension))
    side = np.sqrt(2**(order * dimension)).astype(int)
    coords = np.array(list(map(lambda x :list([x // (side) , x % (side)]), whole_index)))
    return coords

'''
Grid (회색 선) 을 그릴 좌표를 써주는 함수
Arg : pmax 값
'''
def getGridCooridnates(num):
    grid_ticks = np.array([0, 2**num])
    for _ in range(num):
        temp = np.array([])
        for i,k in zip(grid_ticks[0::1], grid_ticks[1::1]):
            if i == 0 :
                temp = np.append(temp, i)
            temp = np.append(temp, (i+k)/2)
            temp = np.append(temp, k)
        grid_ticks = temp
    grid_ticks -= 0.5
    return grid_ticks

def showPoints(data, ax=None, index=True):
    ax = ax or plt.gca()
    pmax = np.ceil(np.log2(np.max(sample_data)))
    pmax = pmax.astype(int)
    offset = 0.5
    cmin = 0
    cmax = 2 ** (pmax) - 1

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
    # plt.show()

    print(f'pmax: {pmax}')
    # return ax

def showlineByIndexorder(data, ax = None, index = True):
    ax = ax or plt.gca()
    if index :
        coordinates = np.array(list(map(lambda x :list([x // (side) , x % (side)]), data)))
    else :
        coordinates = data
    # print(sample_data)
    ax.plot(coordinates[:,0],coordinates[:,1], linewidth=1, linestyle='--')


'''
function changeIndexOrder : 주어진 cell 'a' 와 상하좌우 'b' 에 위치한 cell을 바꿈
* 단, cell 이 모서리 부분에 위치할 경우 특정 signal 을 Agent에게 준다.
* 이때 signal 을 받은 Environment는 종료하지 않고 (done = {still} False), 동시에 큰 negative reward를 준다.
* 0 : up , 1 : down , 2 : left , 3 : right
'''
def changeIndexOrder(indexD, a, b):
    changableCell = -1

    a = a.cpu().numpy().astype(int).item()
    b = b.cpu().numpy().astype(int).item()
    cell = indexD[a][1:].copy()

    if b == 0 : cell[1] += 1   # up
    elif b == 1 : cell[1] -= 1 # down
    elif b == 2 : cell[0] -= 1 # left
    elif b == 3 : cell[0] += 1 # right

    for idx, xC in enumerate(indexD[:,1:3]):
        if xC[0] == cell[0] and xC[1] == cell[1] :
            changableCell = idx

    if changableCell != -1:
        indexD[[a, changableCell]] = indexD[[changableCell, a]]
    return indexD, changableCell

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
    if std == 0 : std = 1
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
        self.first_out = nn.Linear(self.hidden_size, output_size)
        self.second = nn.Linear(self.hidden_size, self.hidden_size)
        self.second_out = nn.Linear(self.hidden_size, 4)

    def forward(self, input):
        output = torch.relu(self.first(input))
        first = self.first_out(output)
        first_output = torch.softmax(first, dim = -1)
        output = torch.relu(self.second(output))
        second = self.second_out(output)
        second_output = torch.softmax(second, dim = -1)
        return first_output, second_output

class Brain():
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions

        self.model = SFCNet(num_states, hidden_size, num_actions)
        if CUDA: self.model.cuda()
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

        self.model.eval()
        output_1, output_2 = self.model(state_in)
        indexes = np.vstack(ep_history[:, 1])
        indexes_1 = torch.from_numpy(indexes[:, 0].astype('Int32')).type(torch.cuda.LongTensor)
        indexes_2 = torch.from_numpy(indexes[:, 1].astype('Int32')).type(torch.cuda.LongTensor)
        reward = torch.from_numpy(ep_history[:, 2].astype('Float32')).type(torch.cuda.FloatTensor)
        responsible_outputs_1 = output_1.gather(1, indexes_1.view(-1, 1))
        responsible_outputs_2 = output_2.gather(1, indexes_2.view(-1, 1))

        # print(loss)
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
            a = np.random.choice(a_dist, p = a_dist)
            a = np.argmax(a_dist == a)
            b_dist = second.cpu().detach().numpy().reshape([-1])
            b = np.random.choice(b_dist, p = b_dist)
            b = np.argmax(b_dist == b)

            return torch.cuda.FloatTensor([[a,b]])

class Agent():
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_policy_function(self, history):
        self.brain.update(history)

    def get_action(self, state, step):
        action = self.brain.decide_action(state, step)
        return action

class Env():
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
    Agent로 부터 행동을 선택하고 그 행동에 맞춰서 이후 상태 관찰,
    관찰에 기반하여 보상을 계산하고 이들을 버퍼에 저장
    버퍼에 충분히 Transition을 저장했다고 생각하면 신경망(Q 함수) 업데이트
    '''

    def run(self):
        episode_10_list = np.zeros(10)
        reward_list = np.zeros(10)

        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[self.data_index] = 1

        h_state = np.concatenate((avail, self.hilbert.getCoords(self.iteration)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(self.iteration)), axis=1)
        h_num = self.analyzer.l2NormLocality(h_state)
        z_num = self.analyzer.l2NormLocality(z_state)

        self.observation = self.reset()
        min_o_num = self.analyzer.l2NormLocality(self.observation)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial : {min_o_num}')

        # 테스트용
        # self.analyzer.l2NormLocality(self.observation)

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            self.observation = self.reset()
            min_o_num = self.analyzer.l2NormLocality(self.observation)
            observation = np.append(self.observation.reshape(1, -1), min_o_num)
            state = torch.from_numpy(observation).type(torch.cuda.FloatTensor)
            state = state.view(1, -1)
            total_sum = 0
            total_reward = 0
            ep_history = []
            done = False

            for step in range(self.MAX_STEP):
                action = self.agent.get_action(state, episode)
                # set_trace()
                self.observation_next, isChanged = self.step(action)
                o_num = self.analyzer.l2NormLocality(self.observation_next)
                total_sum += o_num
                # state에 점 사이 거리 합을 추가함
                observation_next = np.append(self.observation_next.reshape(1, -1), o_num)
                if isChanged != -1 :
                    if min_o_num < o_num:
                        if o_num < min(h_num, z_num):
                            reward = torch.cuda.FloatTensor([0.0])
                        else:
                            done = True
                            reward = torch.cuda.FloatTensor([-10.0])
                    elif min_o_num == o_num:
                        reward = torch.cuda.FloatTensor([0.0])
                    else:
                        reward = torch.cuda.FloatTensor([100.0])
                        min_o_num = o_num
                else :
                    reward = torch.cuda.FloatTensor([-100.0])

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

            episode_10_list = np.hstack((episode_10_list[1:], total_sum / (step + 1)))
            reward_list = np.hstack((reward_list[1:], total_reward))
            print(f'episode {episode} is over within step {step + 1}. '
                      f'Average of the cost in the 10 episodes : {episode_10_list.mean()} And Reward : {reward_list.mean()}')

        with torch.no_grad():
            # 테스트를 위한, max_step 만큼의 action 후 plt 출력
            fig, ax = plt.subplots(1, figsize=(10, 10))
            showPoints(self.data_index, ax)
            # 데이터 활성화 부분을 삭제
            coords = np.delete((state.view(-1)[:-OFFSET]).view(-1, 3).detach().cpu().numpy(), 0, 1)
            showlineByIndexorder(coords, ax, index=False)
        return (state.view(-1)[:-OFFSET]).view(-1, 3).detach().cpu().numpy()

    '''
    주어진 action 을 수행하고 난 뒤의 state를 반환
    '''

    def step(self, choosenAction):
        next_state, isChanged = changeIndexOrder(self.observation, choosenAction[:, 0], choosenAction[:, 1])
        return next_state, isChanged


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
        avail_data = np.array([np.append(x[1:],np.array([i])) for i, x in enumerate(compared_state) if x[0] == 1])
        cost = 0

        for (x,y) in combinations(avail_data, 2):
            dist_2d = np.sum( (x[0:2]-y[0:2])**2 )
            dist_1d = np.abs(x[2]-y[2])
            # Locality Ratio 가 1과 가까운지 측정
            cost += np.abs(1 - (dist_1d/dist_2d))

        return cost

    def sort(self, init, moved):
        # set_trace()
        moved_argsorted = np.lexsort((moved[:, 1], moved[:, 2]))
        init_argsorted = np.lexsort((init[:, 1], init[:, 2]))
        moved[moved_argsorted, 0] = init[init_argsorted, 0]
        return moved


'''
index (n) 은 다음과 같이 좌표로 표시됨
n 의 최댓값은 DIM * ORDER - 1 
좌표 값은 ( n // (DIM * ORDER), n % (DIM * ORDER) ) 
'''

np.random.seed(210)

side = np.sqrt(2**(ORDER*DIM))
scan_index = np.random.choice(2**(DIM*ORDER),size=DATA_SIZE,replace=False)
sample_data = np.array(list(map(lambda x :list([x // (side) , x % (side)]), scan_index)))
# print(sample_index,'\n', sample_data)
fig, ax = plt.subplots(1, figsize=(10,10))
showPoints(sample_data, ax, index=False)
grid_index = np.arange(2**(ORDER*DIM))
showlineByIndexorder(grid_index, ax)

# plt.show()


env = Env(data_index = scan_index, order = ORDER, max_episode = 10000, max_step = 9999, dimension= DIM)
result_state = env.run()

# rl_anal = Analyzer(scan_index, build_init_state(ORDER, DIM), order = ORDER, dim = DIM)
# rl_anal.sumEachPath(result_state)