import sys
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from LSTM_PolicyGradient_rev1 import Net
from LSTM_Sample import init_weights
from itertools import combinations

'''
 * 08-22 : 기존 방식은 locality가 조금이라도 떨어지면 바로 다음 episode로 넘어갔기 때문에 충분히 학습할 여유가 되지않음
            그렇기 때문에 목숨 개념을 추가해서, 최대 x 번까지 locality가 떨어지게 하는 action을 해도 감점만 하고 지나감
            그리고 Agent가 가진 기회가 끝났는지에 대한 내용도 정보에 추가함 
            ( 현재는 True, False로 하는데 이후 그냥 목숨인    양의 정수 값으로 추가할 것인지는 고려 중 에 있음)
            
 * 08-26 : - 점수 부여 방식을 조금 수정했는데, 최소 locality가 action을 통해서 나오지 않더라도, 이전의 locality 보다 났다면,
            조금의 점수를 부여하는 방향으로 바꿨음. 다만 이전의 locality 보다 같으면 감점을 부여함
           - 초기 curve를 zig-zag 뿐만 아니라 hilbert 또는 z-curve로 시작하게끔 수정
           
 * 08-27 : 훈련된 모델을 테스트해볼 수 있도록 test 함수를 Environment 클래스에 추가
 
 * 09-16 : 최근에 수정한 이력이 있음

'''
xrange = range

NOTEBOOK = True
TEST = False
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIM = 2
ORDER = 2
NUM_OF_CELLS = 2 ** (DIM * ORDER)
side = np.sqrt(NUM_OF_CELLS).astype('int')
INDEX_TO_COORDINATE = np.array(list(map(lambda x: list([x // side, x % side]), np.arange(0, NUM_OF_CELLS))))
DATA_SIZE = 10
MAX_EPISODE = 10000
MAX_STEP = 5
INIT_CURVE = 'zig-zag'
LEARNING_RATE = 5e-4  # 학습률
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

    if index:
        coordinates = INDEX_TO_COORDINATE[data]
    else:
        coordinates = data

    ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=1, linestyle='--')

'''
SFC를 만드는 모델

Notice
1. dropout은 쓸지 말지 고민중임
2. embedding vector를 사용할지 말지 고민하고 있음 (각 데이터의 좌표로 유사성을 파악하기)
'''

class SFCNet(Net):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, learning_rate):
        super().__init__(vocab_size, embedding_size, hidden_size, output_size, learning_rate)
        self.a = nn.LSTM(embedding_size + 2, hidden_size, batch_first=True)
        self.GAMMA = 0.0099
        self.DEVICE = DEVICE
        self.optimizer = optim.Adam(lr=learning_rate, params=self.parameters())

    def forward(self, input, avail_id, his=None, action=None):
        embeds = self.emb(input)
        # 활성화된 cell 의 id 에 해당하는 위치에 binary 형태로 활성화 여부를 알림
        avail_binary = torch.zeros(input.size(0), 1).to(DEVICE)
        action_binary = torch.zeros(input.size(0), 1).to(DEVICE)

        temp = input.cpu().clone()

        for x in avail_id:
            avail_binary[np.argwhere(x == temp).item()] = 1
        if len(action) > 0:
            for x in action:
                action_binary[np.argwhere(x == temp).item()] = 1

        embeds = torch.cat((embeds, avail_binary, action_binary), dim =1)

        # cell 교체를 위해 하나를 선택했다면 해당 cell id 에 binary 형태로 알림 (아니면 모두 0로)
        output, (hidden, cell) = self.a(embeds.view(1, len(input), -1), his)
        # output [ :, -1, : ] 와 hidden 값은 같다.
        output = torch.relu(output[:, -1, :])
        output = self.b(output)

        return output, (hidden, cell)


class Env():
    def __init__(self, data_index, max_episode, max_step, init_curve):
        self.MAX_STEP = max_step
        self.MAX_EPISODE = max_episode
        self.data_index = data_index
        self.initial_curve = init_curve

        self.embedding_size = NUM_OF_CELLS * 2
        self.hidden_size = int(NUM_OF_CELLS * 1.5)

        # Reward 설정용
        self.init_coords = build_init_coords(ORDER, DIM, init_curve)
        self.model = SFCNet(vocab_size = NUM_OF_CELLS, embedding_size = self.embedding_size, hidden_size = self.hidden_size,
                            output_size = NUM_OF_CELLS, learning_rate = LEARNING_RATE)
        init_weights(self.model.to(DEVICE))
        self.analyzer = Analyzer(data_index, self.init_coords.copy(), order=ORDER, dim=DIM)
        self.hilbert = HilbertCurve(dimension=DIM)
        self.z = ZCurve(dimension=DIM)

    '''
    초기 state를 생성하는 함수; 
    1. 활성화된 데이터의 binary 표현
    2. query area 단위 따른 clustering 갯수 (max, average) : (미구현) 
    3. 전체 area에서 curve를 따라서 모든 활성화된 데이터를 지날 수 있는 curve의 최소 길이 (또는 query area에서만 구할 수 있는 길이) : (미구현) 
    '''

    def reset(self, data_index):
        avail = np.zeros((NUM_OF_CELLS, 1))
        observation = np.concatenate((avail, INDEX_TO_COORDINATE[data_index]), axis=1)

        return observation

    '''
    Agent로 부터 행동을 선택하고 그 행동에 맞춰서 이후 상태 관찰,
    관찰에 기반하여 보상을 계산하고 이들을 버퍼에 저장
    버퍼에 충분히 Transition을 저장했다고 생각하면 신경망(Q 함수) 업데이트
    '''

    def run(self):
        locality_per_episode_list = []

        avail = np.zeros((NUM_OF_CELLS, 1))

        h_state = np.concatenate((avail, self.hilbert.getCoords(ORDER)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(ORDER)), axis=1)
        h_num = self.analyzer.l2_norm_locality(h_state)
        z_num = self.analyzer.l2_norm_locality(z_state)

        init_order = np.arange(NUM_OF_CELLS)
        observation = self.reset(init_order)
        init_o_num = self.analyzer.l2_norm_locality(observation)
        global_min_o_num = init_o_num  # 최소 locality 의 역(reverse) 값
        global_min_state = np.empty(1)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial ({self.initial_curve}) : {init_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            prev_o_num = -1
            action_list = []

            curr_state_np = np.random.choice(NUM_OF_CELLS, size=(NUM_OF_CELLS), replace=False)
            curr_state_t = torch.from_numpy(curr_state_np).long().to(DEVICE)
            observation = self.reset(curr_state_np)
            min_o_num = self.analyzer.l2_norm_locality(observation)

            # print(curr_state_np)
            curr_his = (Variable(torch.zeros(1, 1, self.hidden_size).to(DEVICE)), Variable(torch.zeros(1, 1, self.hidden_size).to(DEVICE)))
            done = False

            for step in range(self.MAX_STEP * 2):
                logit, his = self.model(input=curr_state_t, avail_id = self.data_index, his = curr_his, action=action_list)
                prob = torch.softmax(logit, dim=-1)
                log_prob = torch.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum()
                a = torch.multinomial(prob.view(-1), num_samples=1).data
                action_list.append(a.item())
                log_prob = log_prob.view(-1).gather(0, Variable(a))

                self.model.entropies.append(entropy)
                self.model.log_probs.append(log_prob)

                # Reward Part
                if len(action_list) <= 1 :
                    reward = 0
                else :
                    curr_state_np[action_list[0]], curr_state_np[action_list[1]] \
                        = curr_state_np[action_list[1]], curr_state_np[action_list[0]]
                    # Action 에 따른 Lc 값 측정
                    observation = self.step(observation, action_list)
                    o_num = self.analyzer.l2_norm_locality(observation)

                    if min_o_num < o_num:
                        done = True
                        reward = -1
                    elif min_o_num == o_num:
                        if action_list[0] == action_list[1]:
                            reward = -1
                        else :
                            reward = -1
                    else:
                        reward = (init_o_num - o_num)
                        min_o_num = o_num

                    action_list = []

                    # Update Minimum reverse of the locality
                    if global_min_o_num > o_num:
                        global_min_o_num = o_num
                        global_min_state = observation.copy()

                self.model.rewards.append(reward)
                curr_state_t = torch.from_numpy(curr_state_np).long().to(DEVICE)
                curr_his = his
                if done :
                    break
            print(f'[GLOB {global_min_o_num:.3f} / INIT {init_o_num:.3f} / EP_NUM {episode}] '
                  f'stop after {step}.. so far [MIN:{min_o_num:.3f}/LAST:{o_num:.3f}/RW {np.mean(self.model.rewards):.3f}]')
            locality_per_episode_list.append(min_o_num)
            # Step For 문 종료
            self.model.update()
            self.model.episode_reset()
        # Episode For 문 종료
        plt.plot(locality_per_episode_list, 'o', markersize=3)
        plt.xlabel('step')
        plt.ylabel('Reverse of the locality')
        plt.axhline(y=init_o_num, color='r', linestyle='-')
        plt.axhline(y=global_min_o_num, color='g', linestyle='-')
        plt.show()

        return global_min_o_num, global_min_state

    def test(self, data_index, max_episode, max_step):
        pass

    '''
    주어진 action 을 수행하고 난 뒤의 state를 반환
    '''

    def step(self, state, action):
        id = []
        # 주어진 action (cell id)에 따른 좌표 값을 찾고 해당 좌표 값에 기반한 id를 찾음
        for x in INDEX_TO_COORDINATE[action]:
            id.append(((state[:, 1] == x[0]) * (state[:, 2] == x[1])).argmax())
        state[[id[0], id[1]]] = state[[id[1], id[0]]]
        return state


'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


class Analyzer():
    def __init__(self, index, init_state, order, dim):
        self.iteration = order
        self.DIM = dim
        self.scan_index = index

        avail = np.zeros((NUM_OF_CELLS, 1))
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

    def l2_norm_locality(self, compared_state):
        for x in INDEX_TO_COORDINATE[self.scan_index]:
            compared_state[((compared_state[:, 1] == x[0]) * (compared_state[:, 2] == x[1])).argmax(), 0] = 1

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

if __name__ == '__main__':
    np.random.seed(210)

    scan_index = np.random.choice(NUM_OF_CELLS, size=DATA_SIZE, replace=False)
    sample_data = INDEX_TO_COORDINATE[scan_index]

    # if NOTEBOOK:
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     showPoints(sample_data, ax, index=False)
    #
    #     if INIT_CURVE == 'hilbert':
    #         showlineByIndexorder(np.array(HilbertCurve(DIM).getCoords(ORDER)), ax, index=False)
    #     elif INIT_CURVE == 'zig-zag':
    #         grid_index = np.arange(NUM_OF_CELLS)
    #         showlineByIndexorder(grid_index, ax)
    #     plt.show(block=True)

    env = Env(data_index=scan_index, max_episode=MAX_EPISODE, max_step=MAX_STEP, init_curve=INIT_CURVE)
    result_value, result_state = env.run()

    print(f'Recorded the minimum reverse of the locality :{result_value}')

    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)
        showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
        plt.show(block=True)

    # Test trained model
    if TEST :
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

    sys.exit(0)