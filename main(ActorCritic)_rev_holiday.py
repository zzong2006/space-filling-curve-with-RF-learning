import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import matplotlib.image as mpimg
from torch.autograd import Variable
from multiprocessing import Process, Pipe

'''
    * 완전 다른 action
'''

xrange = range

NOTEBOOK = True
TEST = False
CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ------------ Curve ------------------------- #
DIM = 2
ORDER = 2
side = np.sqrt(2 ** (ORDER * DIM)).astype('int')
INDEX_TO_COORDINATE = np.array(list(map(lambda x: list([x // side, x % side]), np.arange(0, 2 ** (ORDER * DIM)))), dtype='int')
DATA_SIZE = 5
MAX_EPISODE = 20000
INIT_CURVE = 'hilbert'
NUM_ADVANCED_STEP = 5  # 총 보상을 계산할 때 Advantage 학습을 할 단계 수
# -------- Hyper Parameter --------------- #
LEARNING_RATE = 1e-5  # 학습률
GAMMA = 0.99  # 시간 할인율
ENTROPY_COEFF = 0.001
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 1
OFFSET = 0  # 기존 state 좌표 값 외에 신경망에 추가로 들어갈 정보의 갯수
NUM_PROCESSES = 1  # 동시 실행 환경 수
KERNEL_SIZE = [-1, -1, 2, 3, 4, 4]  # ORDER 가 2 일때 부터 시작하는 kernel size
NUM_CHANNEL = 9




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
        coords = list(map(lambda x: list([x // (side), x % (side)]), whole_index))
    elif init_curve == 'hilbert':
        coords = HilbertCurve(dimension=dimension).getCoords(order=order)
    elif init_curve == 'z':
        coords = ZCurve(dimension=dimension).getCoords(order=order)
    return np.array(coords)


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


class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''

    def __init__(self, num_steps, num_processes, obs_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, NUM_CHANNEL, obs_size, obs_size).to(DEVICE)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(DEVICE)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(DEVICE)
        self.actions = torch.zeros(num_steps, num_processes, 1).long().to(DEVICE)

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


'''
SFC를 만드는 모델

Notice
1. dropout은 쓸지 말지 고민중임
2. embedding vector를 사용할지 말지 고민하고 있음 (각 데이터의 좌표로 유사성을 파악하기)
'''


def init(module, gain):
    '''결합 가중치를 초기화하는 함수'''
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    '''합성곱층의 출력 이미지를 1차원으로 변환하는 층'''

    def forward(self, x):
        return x.view(x.size(0), -1)


class SFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SFCNet, self).__init__()
        cnn_channel = 91
        output_of_cnn = cnn_channel * ((2 ** (ORDER * DIM // 2) - (3 * KERNEL_SIZE[ORDER]) + 3) ** 2)
        self.hidden_size = hidden_size or ((output_of_cnn + output_size) // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNEL, cnn_channel, kernel_size=KERNEL_SIZE[ORDER]),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernel_size=KERNEL_SIZE[ORDER]),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernel_size=KERNEL_SIZE[ORDER]),
            nn.ReLU(),
            Flatten()
        )

        self.input_nn = nn.Linear(output_of_cnn, self.hidden_size)
        self.hidden_nn_1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.hidden_nn_2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.hidden_nn_3 = nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
        self.hidden_nn_4 = nn.Linear(self.hidden_size // 4, self.hidden_size // 4)

        self.actor_nn = nn.Linear(self.hidden_size // 4, output_size)
        self.critic_nn = nn.Linear(self.hidden_size // 4, 1)

    def forward(self, input):
        output = self.conv(input)
        output = torch.relu(self.input_nn(output))
        output = torch.relu(self.hidden_nn_1(output))
        output = torch.relu(self.hidden_nn_2(output))
        output = torch.relu(self.hidden_nn_3(output))
        output = torch.relu(self.hidden_nn_4(output))
        action = self.actor_nn(output)
        value = self.critic_nn(output)

        return action, value


class Brain():
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions
        self.num_states = num_states

        self.model = SFCNet(num_states, hidden_size, num_actions)
        self.model.to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        print(self.model)

    '''
    Policy Gradient 알고리즘으로 신경망의 결합 가중치 학습
    '''

    def update(self, rollouts):
        # Advantage학습의 대상이 되는 5단계 모두를 사용하여 수정
        self.model.eval()

        # 상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산
        obs_shape = rollouts.observations[:-1].size()
        actor_output, values = self.model(rollouts.observations[:-1].view(-1, NUM_CHANNEL, obs_shape[3], obs_shape[4]))
        log_probs = F.log_softmax(actor_output, dim=1)

        action_log_probs = log_probs.gather(1, rollouts.actions.view(-1, 1))

        probs = F.softmax(actor_output, dim=1)

        # 엔트로피 H : action이 확률적으로 얼마나 퍼져 있는가? (비슷한 확률의 다중 액션 -> high, 단일 액션 -> low)
        entropy = -((log_probs * probs).sum(-1).mean())

        action_log_probs = action_log_probs.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)

        # advantage(행동가치(할인 총 보상, discounted reward)-상태가치(critic value)) 계산
        values = values.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
        advantages = rollouts.returns[:-1] - values

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (-action_gain)

        self.model.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록, 경사를 0.5 이하로 제한함(클리핑)

        self.optimizer.step()  # 결합 가중치 수정

    def decide_action(self, state, episode):
        with torch.no_grad():
            action, _ = self.model(state)

            a = torch.softmax(action, dim=1)
            a = a.multinomial(1).data

            # equivalent with ...
            # a, b = np.random.choice(dist, size=2, replace=False, p=dist)
            # a = np.argmax(dist == a)
            # b = np.argmax(dist == b)
            return a

    def compute_value(self, state):
        _, value = self.model(state)
        return value


class Env():
    def __init__(self, data_index, order, max_episode, max_step, init_curve, dimension=2):
        self.MAX_EPISODE = max_episode
        self.data_index = data_index
        self.initial_curve = init_curve

        self.num_action_space = 2 ** (dimension * order)
        self.num_observation_space = np.sqrt(2 ** (dimension * order)).astype('int')

        # Reward 설정용
        self.init_coords = build_init_coords(order, dimension, init_curve)
        self.agent = Brain(self.num_observation_space, self.num_action_space)
        self.analyzer = Analyzer(data_index, self.init_coords.copy(), order=order, dim=dimension)
        self.hilbert = HilbertCurve(dimension=dimension)
        self.z = ZCurve(dimension=dimension)

    '''
    초기 state를 생성하는 함수; 
    '''

    def reset(self, data_index):
        avail = np.zeros((2 ** (ORDER * DIM), 1))
        avail[data_index] = 1
        observation = np.concatenate((avail, self.init_coords), axis=1)
        return observation

    '''
    Agent로 부터 행동을 선택하고 그 행동에 맞춰서 이후 상태 관찰,
    관찰에 기반하여 보상을 계산하고 이들을 버퍼에 저장
    버퍼에 충분히 Transition을 저장했다고 생각하면 신경망(Q 함수) 업데이트
    '''

    def isValid(self, selected_cells, action):
        if selected_cells.shape[0] > 0:
            check = (selected_cells == action)
        else :
            return True

        for i in range(DIM):
            if i == 0 : c = check[:, i]
            else : c *= check[:, i]

        if c.sum() > 0 :
            return False
        else :
            return True

    def run(self):
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()

        order_np = list(np.empty((0,DIM), dtype='int') for _ in range(NUM_PROCESSES))
        obs_np = np.zeros(
            [NUM_PROCESSES, NUM_CHANNEL, self.num_observation_space, self.num_observation_space])  # Numpy 배열
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
        done_np = np.zeros([NUM_PROCESSES, 1])
        o_num = np.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록

        locality_per_step_list = []

        avail = np.zeros((2 ** (ORDER * DIM), 1))

        h_state = np.concatenate((avail, self.hilbert.getCoords(ORDER)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(ORDER)), axis=1)
        h_num = self.analyzer.l2NormLocality(h_state.copy())
        z_num = self.analyzer.l2NormLocality(z_state.copy())

        init_obs = stateMaker(self.data_index, np.empty(0), init=True)
        observation = np.array([stateMaker(self.data_index, np.empty(0), init=True) for i in range(NUM_PROCESSES)])
        grid_coords = np.array([np.concatenate((avail, self.init_coords.copy()), axis=1) for i in range(NUM_PROCESSES)])
        min_o_num = np.array([self.analyzer.l2NormLocality(grid_coords[0].copy())] * NUM_PROCESSES)
        print(f'hilbert : {h_num} , Z : {z_num}, Initial ({self.initial_curve}) : {min(min_o_num)}')

        global_min_o_num = min(min_o_num)  # 최소 locality 의 역(reverse) 값
        global_min_state = np.empty(0)

        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, self.num_observation_space)  # rollouts 객체
        # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
        obs = torch.from_numpy(observation).float().to(DEVICE)
        current_obs = obs
        rollouts.observations[0].copy_(current_obs)
        prev_o_num = None

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action = self.agent.decide_action(rollouts.observations[step], episode)

                # 한 단계를 실행
                for i in range(NUM_PROCESSES):
                    if order_np[i].shape[0] != 2 ** (ORDER * DIM) - 1:
                        if self.isValid(order_np[i], INDEX_TO_COORDINATE[action[i].item()]) :
                            reward_np[i] = 0
                        else :
                            print('실패!', order_np[i], INDEX_TO_COORDINATE[action[i].item()])
                            reward_np[i] = - (2 ** (DIM * ORDER) / (order_np[i].shape[0]))
                            done_np[i] = True

                    obs_np[i] = stateMaker(self.data_index, order_np[i],
                                           action=action[i].item(),
                                           init_state=init_obs)

                    if not done_np[i].item():
                        order_np[i] = np.vstack( (order_np[i], INDEX_TO_COORDINATE[action[i]].reshape(1,-1)))

                    if order_np[i].shape[0] == 2 ** (ORDER * DIM) :
                        o_num[i] = self.analyzer.l2NormLocality(np.concatenate((avail, order_np[i]), axis=1))
                        print('완주!', order_np, INDEX_TO_COORDINATE[action[i]], o_num[i])
                        if min_o_num[i] < o_num[i]:
                            reward_np[i] = 1
                        elif min_o_num[i] == o_num[i]:
                            reward_np[i] = 5
                        else:
                            reward_np[i] = 100
                            min_o_num[i] = o_num[i]
                        order_np[i] = np.empty((0,DIM), dtype='int')

                    # o_num[i] = self.analyzer.l2NormLocality(obs_next.copy())
                    # if i == 0:
                    #     # print(action[i][0].item(), action[i][1].item())
                    #     locality_per_step_list.append(o_num[i])
                    #     writer.add_scalar('Result/Lc', o_num[i], (episode + step))
                    #
                    # # Update Minimum reverse of the locality
                    # if global_min_o_num > o_num[i]:
                    #     global_min_o_num = o_num[i]
                    #     global_min_state = obs_next.copy()

                reward = torch.from_numpy(reward_np).float().to(DEVICE)
                # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])

                current_obs = torch.from_numpy(obs_np).float().to(DEVICE)

                rollouts.insert(current_obs, action.data, reward, masks)

                for i in range(NUM_PROCESSES):
                    if done_np[i] :
                        done_np[i] = False
                        order_np[i] = np.empty((0,DIM), dtype='int')

                # advanced 학습 for문 끝
            # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산
            with torch.no_grad():
                next_value = self.agent.compute_value(rollouts.observations[-1]).detach()
            rollouts.compute_returns(next_value)

            # 신경망 및 rollout 업데이트
            self.agent.update(rollouts)
            rollouts.after_update()

            print(f'{episode} : {global_min_o_num}')

        if NOTEBOOK:
            plt.plot(locality_per_step_list, 'b-')
            plt.xlabel('step')
            plt.ylabel('Reverse of the locality')

            plt.tight_layout()
            plt.show(block=True)

        return global_min_o_num, global_min_state

    '''
    주어진 action 을 수행하고 난 뒤의 state를 반환
    '''

    def step(self, state, choosenAction):
        state = np.vstack(state, INDEX_TO_COORDINATE[choosenAction])
        return state


'''
 08-29 : 주어진 grid id 순서에 따라 coords를 이용하여 images 생성
            총 7가지 채널을 만드는것을 목표로 함
  ARGS : data_index (현재 curve에 따른 grid의 순서를 index 화 시켜서 나타낸 것)
'''


def stateMaker(data_index, ordered_coords, action=None, init=False, init_state=None):
    data_coords = INDEX_TO_COORDINATE[data_index]
    input_state = np.zeros([NUM_CHANNEL, side, side])

    if init:
        # 활성화된 데이터
        input_state[0][data_coords[:, 0], data_coords[:, 1]] = 1

        # 모든 이미지가 0, 모든 이미지가 1
        input_state[2][:] = 1

        # 활성화 되지 않은 grid
        input_state[4][:] = 1 - input_state[0]
    else:
        input_state = init_state.copy()

    # 현재 선택한 cell 위치
    if type(action) is int:
        choosed_location = INDEX_TO_COORDINATE[action]
        input_state[3][choosed_location[0], choosed_location[1]] = 1

    # 지금까지 선택한 cell (현재 선택한 cell 제외)
    if ordered_coords.shape[0] != 0:
        input_state[5][ordered_coords[:, 0], ordered_coords[:, 1]] = 1

    # 지금까지 선택한 cell (현재 선택한 cell 포함)
    input_state[6] = input_state[3]
    if ordered_coords.shape[0] != 0:
        input_state[6][ordered_coords[:, 0], ordered_coords[:, 1]] = 1

    # 선택 가능한 cell
    input_state[7] = 1 - input_state[6]

    # 지금까지 선택한 cell 의 순서 값을 오름차순(0 ~ 0.9)으로 나타냄. 그리고 나머지 cell 들의 값은 1로 채움
    input_state[8][:] = 1
    if ordered_coords.shape[0] != 0:
        order_value = np.linspace(0, 0.9, num=ordered_coords.shape[0])
        input_state[8][ordered_coords[:, 0], ordered_coords[:, 1]] = order_value[::-1]
    if type(action) is int:
        choosed_location = INDEX_TO_COORDINATE[action]
        input_state[8][choosed_location[0], choosed_location[1]] = 0.95

    return input_state


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

        # zig-zag curve 를 기준으로 활성화 데이터의 좌표계를 만든다
        # 다른 curve 로 만드려면 locality 측정에 혼선이 있을 수 있음 (실제로 값이 다름)
        side = np.sqrt(2 ** (ORDER * DIM)).astype('int32')
        coords = np.array(list(map(lambda x: list([x // (side), x % (side)]), np.arange(2 ** (order * dim)))))

        self.init_state = np.concatenate((avail, coords), axis=1)

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


def main():
    np.random.seed(210)

    scan_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
    sample_data = np.array(INDEX_TO_COORDINATE[scan_index])

    # 초기 curve 와 활성화 데이터 plot
    if NOTEBOOK and False:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)

        if INIT_CURVE == 'hilbert':
            showlineByIndexorder(np.array(HilbertCurve(DIM).getCoords(ORDER)), ax, index=False)
        elif INIT_CURVE == 'zig-zag':
            grid_index = np.arange(2 ** (ORDER * DIM))
            showlineByIndexorder(grid_index, ax)
        plt.show(block=True)

    env = Env(data_index=scan_index, order=ORDER, max_episode=MAX_EPISODE, max_step=10,
              init_curve=INIT_CURVE, dimension=DIM)
    result_value, result_state = env.run()

    print(f'Recorded the minimum reverse of the locality :{result_value}')
    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, ax, index=False)
        showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
        plt.show(block=True)

    # Test trained model
    if TEST:
        np.random.seed(175)

        print(f'Start testing trained model ... ')
        test_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
        result_value, result_state = env.test(test_index, max_episode=500, max_step=100)

        print(f'[TEST] Recorded the minimum reverse of the locality :{result_value}')

        if NOTEBOOK:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            showPoints(sample_data, ax, index=False)
            showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
            plt.show(block=True)
    return 0


if __name__ == '__main__':
    main()
