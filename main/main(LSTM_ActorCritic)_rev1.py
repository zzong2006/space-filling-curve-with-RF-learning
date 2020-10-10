import sys
from itertools import combinations

import torch
import torch.nn as nn
from torch import optim
from utils import *

import cherry as ch
from cherry import pg
from torch.distributions import Normal

xrange = range

torch.manual_seed(123456789)
NOTEBOOK = False
TEST = False
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIM = 2
ORDER = 2
NUM_OF_CELLS = 2 ** (DIM * ORDER)
side = (int)(np.sqrt(NUM_OF_CELLS))
INDEX_TO_COORDINATE = np.array(list(map(lambda x: list([x // side, x % side]), np.arange(0, NUM_OF_CELLS))))
DATA_SIZE = 5
MAX_EPISODE = 10000
TASKS_PER_META_BATCH = 1000
HORIZON = 32
INIT_CURVE = 'zig-zag'

# Hyper-Parameters
DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
MAX_STEPS = 500
BATCH_SIZE = 2048
TRACE_DECAY = 0.97
SEED = 42
PPO_CLIP_RATIO = 0.2
PPO_EPOCHS = 20
REPLAY_SIZE = 100000

'''
SFC를 만드는 모델
'''
class SFCNet(nn.Module):
    def __init__(self, embedding_size, hidden_size, linear_hidden, learning_rate):
        super(SFCNet, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.actor = nn.Sequential(*[
            nn.Linear(hidden_size + (DIM + 1), linear_hidden),
            nn.Tanh(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.Tanh(),
            nn.Linear(linear_hidden, 1)
        ])
        self.critic = nn.Sequential(*[
            nn.Linear(hidden_size + (DIM + 1), linear_hidden),
            nn.Tanh(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.Tanh(),
            nn.Linear(linear_hidden, 1)
        ])

        self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

        self.avail_index = []
        self.hidden_size = hidden_size

        self.DEVICE = DEVICE
        self.log_std_min, self.log_std_max = -20, 2
    '''
    input : time-step
    '''
    def forward(self, input, coord, his=None, preprocess = True):
        if preprocess :
            if input.size() == torch.Size([1, 1, DIM + 2]) and input.sum() == 0:
                output = torch.zeros(self.hidden_size).to(DEVICE)
                his = (torch.zeros(1, 1, self.hidden_size).to(DEVICE),
                       torch.zeros(1, 1, self.hidden_size).to(DEVICE))
                preprocessed_output = torch.cat((output.view(-1), coord))
            else :
                output, his = self.lstm(input, his)
                # output [ :, -1, : ] 와 hidden 값은 같다.
                output = torch.tanh(his[0])
                preprocessed_output = torch.cat((output.view(-1), coord))
        else :
            preprocessed_output = input

        policy = Normal(self.actor(preprocessed_output), self.policy_log_std.exp())
        value = self.critic(preprocessed_output)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {
            'after_lstm' : preprocessed_output,
            'mass' : policy,
            'log_prob' : log_prob,
            'value' : value,
            'hidden' : his
        }

    def update(self, em):
        #self.optimizer.zero_grad()
        loss = self.conjugate_loss(memory=em)
        print(loss)
        #loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
        #self.optimizer.step()

    def conjugate_loss(self, memory):
        optimizer = optim.Adam(params = self.parameters(), lr=LEARNING_RATE)
        critic_optimizer = optim.Adam(params = self.critic.parameters(), lr=LEARNING_RATE)
        with torch.no_grad():
            advantages = pg.generalized_advantage(DISCOUNT, TRACE_DECAY,
                                                  memory.reward(),
                                                  memory.done(),
                                                  memory.value(),
                                                  torch.zeros(1).to(DEVICE))
            advantages = ch.normalize(advantages, epsilon=1e-8)
            returns = ch.td.discount(DISCOUNT, memory.reward(), memory.done())
            old_log_probs = memory.log_prob()
        new_values = memory.value()
        new_log_probs = memory.log_prob()
        for epoch in range(PPO_EPOCHS):
            if epoch > 0 :
                _, infos = self(memory.state(), coord = None, his= memory.his(), preprocess = False)
                masses = infos['mass']
                new_values = infos['value'].view(-1, 1)
                new_log_probs = masses.log_prob(memory.action())

            policy_loss = ch.algorithms.ppo.policy_loss(new_log_probs, old_log_probs,
                                                        advantages, clip=PPO_CLIP_RATIO)

            # actor_optimizer.zero_grad()
            # policy_loss.backward(retain_graph=True)
            # actor_optimizer.step()

            value_loss = ch.algorithms.a2c.state_value_loss(new_values, returns)

            optimizer.zero_grad()
            if epoch != PPO_EPOCHS -1 :
                (policy_loss + value_loss).backward(retain_graph=True)
            else :
                (policy_loss + value_loss).backward()
            optimizer.step()

            # print(self.state_dict())
        return (policy_loss + value_loss)



class EpisodeMemory():
    def __init__(self):
        self.value_list = []
        self.log_prob_list = []
        self.reward_list = []
        self.entropy_list = []

    def append(self, value, log_prob, reward, entropy):
        self.value_list.append(value)
        self.log_prob_list.append(log_prob)
        self.reward_list.append(reward)
        self.entropy_list.append(entropy)

    def episode_reset(self):
        self.value_list = []
        self.log_prob_list = []
        self.reward_list = []
        self.entropy_list = []

class Env():
    '''
    embedding size = ( dim coord , avail binary, continuous value )
    ex ) dim = 2 -> embedding size = 2 + 2 = 4
    '''
    def __init__(self, data_index, max_episode):
        self.MAX_EPISODE = max_episode
        self.data_index = data_index

        embedding_size = DIM + 2
        self.hidden_size = side

        # Reward 설정용
        self.model = SFCNet(embedding_size= embedding_size, hidden_size= side,
                            linear_hidden= side // 2, learning_rate=LEARNING_RATE)

        self.model.to(DEVICE)
        self.hilbert = HilbertCurve(dimension=DIM)
        self.z = ZCurve(dimension=DIM)

    @staticmethod
    def reset(data_index):
        avail = np.zeros((NUM_OF_CELLS, 1))
        observation = np.concatenate((avail, INDEX_TO_COORDINATE[data_index]), axis=1)

        return observation

    def make_task(self, state):
        avail_index = np.random.choice(NUM_OF_CELLS, size=DATA_SIZE, replace=False)
        obs = self.reset(state)
        return avail_index, obs


    def run(self):
        print(self.model)

        locality_list_per_episode = []
        value_list_per_episode = []
        accumulated_reward_per_episode = []

        self.model.avail_index = self.data_index.copy()
        self.model.done = True

        analyzer = Analyzer(self.data_index)
        avail = np.zeros((NUM_OF_CELLS, 1))
        h_state = np.concatenate((HilbertCurve(dimension=DIM).getCoords(ORDER), avail), axis=1)
        z_state = np.concatenate((ZCurve(dimension=DIM).getCoords(ORDER), avail), axis=1)
        h_num = analyzer.l2_norm_locality(h_state)
        z_num = analyzer.l2_norm_locality(z_state)
        
        # 초기에 랜덤으로 continuous action 을 2 ** (order * dim) 만큼 생성해서 완성된 curve의 locality를 측정
        grid_info = torch.zeros((1,1, DIM + 2)).to(DEVICE)
        with torch.no_grad() :
            his = (torch.zeros(1, 1, self.hidden_size).to(DEVICE),
             torch.zeros(1, 1, self.hidden_size).to(DEVICE))
            cursor = 0
            for i in range(NUM_OF_CELLS):
                x, y = INDEX_TO_COORDINATE[i]
                if DATA_SIZE > cursor and self.data_index[cursor] == i :
                    avail = 1
                    cursor += 1
                else :
                    avail = 0
                coord = torch.FloatTensor([x, y, avail]).to(DEVICE)
                action, state = self.model(input= grid_info, coord = coord, his = his)
                his = state['hidden']

                if grid_info.size() == torch.Size([1,1,DIM + 2]) and grid_info.sum() == 0:
                    grid_info = torch.FloatTensor([[[x, y, avail, action]]]).to(DEVICE)
                else :
                    grid_info = torch.cat( (grid_info, torch.FloatTensor([[[x, y, avail, action]]]).to(DEVICE)) , axis=1)

        observation = grid_info.squeeze_(dim=0).cpu().numpy()
        # 생성된 continuous action 을 기반으로 정렬함
        print(observation[observation[:,3].argsort()])
        init_o_num = analyzer.l2_norm_locality(observation, continuous=True)
        global_min_o_num = init_o_num  # 최소 locality 의 역(reverse) 값
        
        print(f'hilbert : {h_num} , Z : {z_num}, initial_curve : {init_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            replay = self.run_model(self.model, self.data_index)

            # Step For 문 종료
            self.model.update(replay)
            replay.empty()
        # Episode For 문 종료

        plt.plot(locality_list_per_episode, 'o', markersize=2)
        plt.xlabel('step')
        plt.ylabel('Invert of the locality')
        plt.axhline(y=init_o_num, color='r', linestyle='-')
        plt.axhline(y=global_min_o_num, color='g', linestyle='-')
        plt.show()

        plt.plot(value_list_per_episode, 'o', markersize=2)
        plt.show()

        return global_min_o_num
    '''
    주어진 모델을 이용해서 continuous action을 수행 후, 모은 정보에 대한 replay 메모리를 반환
    input : model, 활성화 인덱스
    output : replay memory(buffer)
    '''
    def run_model(self, model, data_index):
        replay = ch.ExperienceReplay()
        analyzer = Analyzer(data_index)

        grid_info = torch.zeros((1,1,DIM + 2)).to(DEVICE)
        his = (torch.zeros(1, 1, self.hidden_size).to(DEVICE),
             torch.zeros(1, 1, self.hidden_size).to(DEVICE))
        cursor = 0
        for i in range(NUM_OF_CELLS):
            x, y = INDEX_TO_COORDINATE[i]
            # 활성화된 데이터를 검색 (cursor index 사용)
            if DATA_SIZE > cursor and data_index[cursor] == i:
                avail = 1
                cursor += 1
            else:
                avail = 0
            coord = torch.FloatTensor([x, y, avail]).to(DEVICE)
            action, state = model(input=grid_info, coord=coord, his=his)

            # 필요한 정보 : reward, done, value, log_prob, action, his
            if i != NUM_OF_CELLS - 1 :
                replay.append(state=state['after_lstm'], action=action, reward=0, next_state=torch.zeros(1), done=False,
                    log_prob = state['log_prob'],
                    value = state['value'],
                    his = his
                )
                his = state['hidden']

            if grid_info.size() == torch.Size([1,1,DIM + 2]) and grid_info.sum() == 0:
                grid_info = torch.FloatTensor([[[x, y, avail, action]]]).to(DEVICE)
            else:
                grid_info = torch.cat((grid_info, torch.FloatTensor([[[x, y, avail, action]]]).to(DEVICE)), axis=1)

        observation = grid_info.squeeze_(dim=0).cpu().numpy()
        # 생성된 continuous action 을 기반으로 정렬함
        observation = observation[observation[:, 3].argsort()]
        # print(observation[observation[:, 3].argsort()])
        # continuous actions 을 통해 curve를 형성했다면 locality를 측정한다.
        o_num = analyzer.l2_norm_locality(observation, continuous=True)

        # reward 가 부여된 마지막 replay content append
        print('reward : ' , (1 - o_num), ' o_num : ', o_num)
        replay.append(state=state['after_lstm'], action=action, reward=(1 - o_num), next_state=torch.zeros(1), done=True,
                    log_prob = state['log_prob'],
                    value = state['value'],
                    his = his
        )

        return replay.to(DEVICE)
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

    # def give_reward(self, min_num, num):
    #     if min_num < num:
    #         reward = num
    #     elif min_num == num:
    #         reward = -0.01
    #     else:
    #         reward = (min_num - num)
    #     return reward
'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


class Analyzer():
    def __init__(self, index, init_state=None):
        self.scan_index = index

        avail = np.zeros((NUM_OF_CELLS, 1))
        avail[index] = 1
        if init_state is not None :
            self.init_state = np.concatenate((avail, init_state), axis=1)

    def l2_norm_locality(self, compared_state, continuous=False):
        if not continuous:
            for x in INDEX_TO_COORDINATE[self.scan_index]:
                compared_state[((compared_state[:, 0] == x[0]) * (compared_state[:, 1] == x[1])).argmax(), 2] = 1

        # 활성화된 데이터만 모음, 결과는 (x, y, 데이터 순서)
        avail_data = np.array([np.append(x[0:2], np.array([i])) for i, x in enumerate(compared_state) if x[2] == 1])
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
    np.random.seed(210) # original : 210

    scan_index = np.sort(np.random.choice(NUM_OF_CELLS, size=DATA_SIZE, replace=False))
    sample_data = INDEX_TO_COORDINATE[scan_index]
    print(f'Avail data set {scan_index}')

    if NOTEBOOK:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        showPoints(sample_data, order = ORDER, dim = DIM,  ax = ax, index=False)

        if INIT_CURVE == 'hilbert':
            showlineByIndexorder(np.array(HilbertCurve(DIM).getCoords(ORDER)), INDEX_TO_COORDINATE, ax, index=False)
        elif INIT_CURVE == 'zig-zag':
            grid_index = np.arange(NUM_OF_CELLS)
            showlineByIndexorder(grid_index, INDEX_TO_COORDINATE, ax)
        plt.show(block=True)

    env = Env(data_index=scan_index, max_episode=MAX_EPISODE)

    result_value = env.run()



    print(f'Recorded the minimum reverse of the locality :{result_value}')

    # if NOTEBOOK:
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     showPoints(sample_data, ax, index=False)
    #     showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
    #     plt.show(block=True)

    # Test trained model
    if TEST:
        np.random.seed(175)

        print(f'Start testing trained model ... ')
        test_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
        print(f'Avail data set {test_index}')
        env.data_index = test_index
        result_value = env.run()

        print(f'[TEST]Recorded the minimum reverse of the locality :{result_value}')

    sys.exit(0)