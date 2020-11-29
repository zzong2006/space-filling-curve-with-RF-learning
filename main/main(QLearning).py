import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from curve import HilbertCurve, ZCurve

CUDA = torch.cuda.is_available()
DIM = 2
ORDER = 4
DATA_SIZE = 10
BATCH_SIZE = 32
OFFSET = 1
MAX_STEP = 200
CAPACITY = 10000
GAMMA = 0.99  # 시간 할인율

Transition = namedtuple('Trainsition', ('state', 'action', 'next_state', 'reward'))

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


def showlineByIndexorder(data, ax=None, index=True):
    ax = ax or plt.gca()
    if index:
        coordinates = np.array(list(map(lambda x: list([x // (side), x % (side)]), data)))
    else:
        coordinates = data
    # print(sample_data)
    ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=1, linestyle='--')


def changeIndexOrder(indexD, a, b):
    a = a.cpu().numpy().astype(int).item()
    b = b.cpu().numpy().astype(int).item()
    # set_trace()
    indexD[[a, b]] = indexD[[b, a]]
    return indexD


'''
SFC를 만드는 모델

Notice
1. dropout은 쓸지 말지 고민중임
2. gru는 하나만 만들고 for문으로 돌려서 쓰는것 (아무래도 2번만 필요하니까 range(2) 일 듯)
3. embedding vector를 사용할지 말지 고민하고 있음 (각 데이터의 좌표로 유사성을 파악하기)
'''


class SFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SFCNet, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.first = nn.Linear(input_size, self.hidden_size)
        self.first_relu = nn.ReLU()
        self.first_out = nn.Linear(self.hidden_size, output_size)
        self.second = nn.Linear(self.hidden_size, self.hidden_size)
        self.second_relu = nn.ReLU()
        self.second_out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        output = self.first_relu(self.first(input))
        first_output = self.first_out(output)
        output = self.second_relu(self.second(output))
        second_output = self.second_out(output)
        return first_output, second_output


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 메모리 최대 저장 건수
        self.memory = []  # 실제 transition을 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수

    '''
    transition = (state, action, state_next, reward)을 메모리에 저장
    '''

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Transition이라는 namedtuple을 사용해 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    '''
    batch_size 개수만큼 무작위로 저장된 transition 호출
    '''

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    '''
    len 함수로 현재 저장된 transition 개수를 반환
    '''

    def __len__(self):
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions, hidden_size=None):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = SFCNet(num_states, hidden_size, num_actions)
        if CUDA: self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        print(self.model)

    '''
    Experience Replay로 신경망의 결합 가중치 학습
    '''

    def replay(self):
        # 저장된 transition 수가 mini_batch 크기보다 작으면 아무 것도 안함
        if len(self.memory) < BATCH_SIZE:
            return

        # 미니배치 생성
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states_batch = torch.cat(batch.next_state)
        # 정답 신호로 사용할 Q(s_t, a_t)를 계산
        self.model.eval()
        # set_trace()
        first, second = self.model(state_batch)
        first_state_action_values = first.gather(1, action_batch[:, 0].view(-1, 1))
        second_state_action_values = second.gather(1, action_batch[:, 1].view(-1, 1))

        # max{Q(s_{t+1}, a)} 값을 계산
        first, second = self.model(next_states_batch)

        first_next_state_values = first.max(1)[0].detach()
        second_next_state_values = second.max(1)[0].detach()

        # 정답 신호로 사용할 Q(s_t, a_t)값을 Q 러닝으로 계산
        # SAV : State Action Values

        first_expected_SAV = reward_batch + GAMMA * first_next_state_values
        second_expected_SAV = reward_batch + GAMMA * second_next_state_values

        self.model.train()
        # smooth_l1_loss는 Huber 함수
        loss_1 = F.smooth_l1_loss(first_state_action_values, first_expected_SAV.unsqueeze(1))
        loss_2 = F.smooth_l1_loss(second_state_action_values, second_expected_SAV.unsqueeze(1))
        total_loss = loss_1 + loss_2
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        epsilon = 0.5 * (1 / np.log2(episode + 1 + 1e-7))

        if epsilon < np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                first, second = self.model(state.view(1, -1))
                return torch.cat((first, second)).max(1)[1].view(1, 2)
        else:
            action = np.random.choice(self.num_actions, size=(1, 2))
            action = torch.from_numpy(action).type(torch.cuda.LongTensor)
        return action


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, step):
        action = self.brain.decide_action(state, step)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)



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
        avail = np.zeros((2 ** (self.iteration * self.DIM), 1))
        avail[self.data_index] = 1
        h_state = np.concatenate((avail, self.hilbert.getCoords(self.iteration)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(self.iteration)), axis=1)
        h_num = self.analyzer.sumEachPath(h_state)
        z_num = self.analyzer.sumEachPath(z_state)
        self.observation = self.reset()
        min_o_num = self.analyzer.sumEachPath(self.observation)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial : {min_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            training_final = False
            self.observation = self.reset()
            min_o_num = self.analyzer.sumEachPath(self.observation)
            observation = np.append(self.observation.reshape(1, -1), min_o_num)
            state = torch.from_numpy(observation).type(torch.cuda.FloatTensor)
            state = state.view(1, -1)
            total_sum = 0
            done = False

            for step in range(self.MAX_STEP):
                action = self.agent.get_action(state, episode)
                # set_trace()
                self.observation_next = self.step(action)
                o_num = self.analyzer.sumEachPath(self.observation_next)
                total_sum += o_num
                # state에 점 사이 거리 합을 추가함
                observation_next = np.append(self.observation_next.reshape(1, -1), o_num)

                if min_o_num < o_num:
                    if o_num < min(h_num, z_num):
                        reward = torch.cuda.FloatTensor([0.0])
                    else:
                        done = True
                        reward = torch.cuda.FloatTensor([-1.0])
                elif min_o_num == o_num:
                    reward = torch.cuda.FloatTensor([0.0])
                else:
                    reward = torch.cuda.FloatTensor([1.0])
                    min_o_num = o_num

                state_next = observation_next
                state_next = torch.from_numpy(state_next).type(torch.cuda.FloatTensor)
                state_next = state_next.view(1, -1)
                # set_trace()
                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next

                if done:
                    break
            episode_10_list = np.hstack((episode_10_list[1:], total_sum / (step + 1)))
            print(
                f'episode {episode} is over within step {step + 1}. Average of the cost in the 10 episodes : {episode_10_list.mean()} ')

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
        # set_trace()
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

np.random.seed(150)

side = np.sqrt(2 ** (ORDER * DIM))
scan_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
sample_data = np.array(list(map(lambda x: list([x // (side), x % (side)]), scan_index)))
# print(sample_index,'\n', sample_data)
fig, ax = plt.subplots(1, figsize=(10, 10))
showPoints(sample_data, ax, index=False)
grid_index = np.arange(2 ** (ORDER * DIM))
showlineByIndexorder(grid_index, ax)

plt.show()

env = Env(data_index=scan_index, order=ORDER, max_episode=1000, max_step=1000, dimension=DIM)
result_state = env.run()

rl_anal = Analyzer(scan_index, build_init_state(ORDER, DIM), order=ORDER, dim=DIM)
rl_anal.sumEachPath(result_state)
