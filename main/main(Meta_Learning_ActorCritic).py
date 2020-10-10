import sys
from itertools import combinations

import learn2learn as l2l
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
from reinforcement_learning_review.LSTM_Sample import init_weights
from utils import *

'''
    * Meta-Learning : MAML 적용 (단일 노드로 시작, task 는 같은 DATA_SIZE 에서 랜덤한 배치로만 )
'''
xrange = range

torch.manual_seed(123456789)
NOTEBOOK = True
TEST = False
LOAD_AND_SAVE = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

##### Curve 정보
DIM = 2
ORDER = 2
DATA_SIZE = 5
INIT_CURVE = 'zig-zag'

##### 일반 학습
MAX_EPISODE = 10000
MAX_STEP = 5
LEARNING_RATE = 0.0005  # 학습률

##### Meta 학습
## 총 meta learning 횟수 : META_STEP * #_of_TASKS * Horizon
META_LEARNING_STEP = 1000        # 메타 learning 횟수
TASKS_PER_META_BATCH =  20       # 메타 학습 1회 당 생성할 총 task 의 갯수
HORIZON = 32                    # task 1개 당 수행할 action step 수
META_LEARNING_RATE = 0.005  # 메타 학습률

NUM_OF_CELLS = 2 ** (DIM * ORDER)
side = np.sqrt(NUM_OF_CELLS).astype('int')
INDEX_TO_COORDINATE = np.array(list(map(lambda x: list([x // side, x % side]), np.arange(0, NUM_OF_CELLS))))
'''
SFC를 만드는 모델
'''


class SFCNet(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, learning_rate):
        super(SFCNet, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + 2, hidden_size, batch_first=True)
        self.critic_hidden_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_hidden_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_hidden_2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.actor_hidden_2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.critic_linear = nn.Linear(hidden_size // 4, 1)
        self.actor_linear = nn.Linear(hidden_size // 4, output_size)

        self.avail_index = []
        self.his = (torch.zeros(1, 1, hidden_size).to(DEVICE), torch.zeros(1, 1, hidden_size).to(DEVICE))
        self.state = np.arange(0, NUM_OF_CELLS)
        self.hidden_size = hidden_size
        self.done = True
        self.GAMMA = 0.999
        self.tau = 1.00
        self.DEVICE = DEVICE


    def forward(self, input, avail_id, his=None, action=None):

        flatten_input = input.flatten()
        embeds = self.emb(flatten_input)
        # 활성화된 cell 의 id 에 해당하는 위치에 binary 형태로 활성화 여부를 알림
        avail_binary = torch.zeros(flatten_input.size(0), 1).to(DEVICE)
        action_binary = torch.zeros(flatten_input.size(0), 1).to(DEVICE)

        temp = flatten_input.cpu().clone()

        for x in avail_id:
            avail_binary[np.argwhere((x == temp)).item()] = 1
        if len(action) > 0:
            for x in action:
                action_binary[np.argwhere((x == temp).flatten()).item()] = 1

        embeds = torch.cat((embeds, avail_binary, action_binary), dim=1)
        self.lstm.flatten_parameters()
        # cell 교체를 위해 하나를 선택했다면 해당 cell id 에 binary 형태로 알림 (아니면 모두 0로)
        output, hn = self.lstm(embeds.view(1, len(flatten_input), -1), his)

        # output [ :, -1, : ] 와 hidden 값은 같다.
        output = torch.relu(output[:, -1, :])
        act_output = torch.relu(self.actor_hidden_1(output))
        act_output = torch.relu(self.actor_hidden_2(act_output))
        act = self.actor_linear(act_output)
        critic_output = torch.relu(self.critic_hidden_1(output))
        critic_output = torch.relu(self.critic_hidden_2(critic_output))
        critic = self.critic_linear(critic_output)
        return critic, act, hn

    def update(self, em, optimizer):
        optimizer.zero_grad()
        loss = self.conjugate_loss(em)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    def conjugate_loss(self, memory):
        # next_value, mass, his = self(memory[-1].next_state, avail_id = self.avail_index, his=self.his, action=[])
        # advantages = ch.pg.generalized_advantage(0.99,
        #                                    0.95,
        #                                    memory.reward(),
        #                                    memory.value(),
        #                                    memory.done(),
        #                                    next_value.detach())
        # policy_loss = ch.algorithms.a2c.policy_loss(memory.log_prob(), advantages)
        # value_loss = ch.algorithms.a2c.state_value_loss(memory.value(),memory.reward())

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(DEVICE)
        R = torch.zeros(1, 1).to(DEVICE)

        if not self.done:
            curr_state_t = torch.from_numpy(self.state).long().to(DEVICE)
            value, _, _ = self(input=curr_state_t, avail_id=self.avail_index, his=self.his, action=[])
            R = value.data

        memory.value_list.append(R)
        episode_length = len(memory.reward_list)

        for i in reversed(range(episode_length)):
            R = self.GAMMA * R + memory.reward_list[i]
            # advantage = R - self.values[i]
            value_loss = value_loss + F.smooth_l1_loss(R, memory.value_list[i])

            # Generalized Advantage Estimation
            delta_t = memory.reward_list[i] + self.GAMMA * memory.value_list[i + 1].data - memory.value_list[i].data
            gae = gae * self.GAMMA * self.tau + delta_t

            policy_loss -= (gae * memory.log_prob_list[i] - 0.01 * memory.entropy_list[i])
        total_loss = policy_loss + 0.5 * value_loss

        return total_loss


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
    def __init__(self, data_index, max_episode, max_step, init_curve):
        self.MAX_STEP = max_step
        self.MAX_EPISODE = max_episode
        self.data_index = data_index
        self.initial_curve = init_curve

        self.embedding_size = NUM_OF_CELLS * 2
        self.hidden_size = int(NUM_OF_CELLS * 1.5)

        # Reward 설정용
        self.init_coords = build_init_coords(ORDER, DIM, init_curve)
        self.model = SFCNet(vocab_size=NUM_OF_CELLS, embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                            output_size=NUM_OF_CELLS, learning_rate=LEARNING_RATE)

        init_weights(self.model.to(DEVICE))
        self.hilbert = HilbertCurve(dimension=DIM)
        self.z = ZCurve(dimension=DIM)

        print(self.model)

    '''
    초기 state를 생성하는 함수; 
    2. query area 단위 따른 clustering 갯수 (max, average) : (미구현) 
    '''

    @staticmethod
    def reset(data_index):
        avail = np.zeros((NUM_OF_CELLS, 1))
        observation = np.concatenate((avail, INDEX_TO_COORDINATE[data_index]), axis=1)

        return observation

    def make_task(self, state):
        avail_index = np.random.choice(NUM_OF_CELLS, size=DATA_SIZE, replace=False)
        obs = self.reset(state)
        return avail_index, obs

    def meta_run(self):
        optimizer = optim.Adam(lr=META_LEARNING_RATE, params=self.model.parameters())
        maml = l2l.algorithms.MAML(self.model, lr=0.001)

        for iteration in range(META_LEARNING_STEP):  # meta-learning 횟수
            grad = None
            print('Meta Learning Progress : {0}%'.format(iteration / META_LEARNING_STEP * 100))
            for i in range(TASKS_PER_META_BATCH):  # sampling task 횟수
                curr_state_np = np.arange(0, NUM_OF_CELLS)
                avail_idx, obs = self.make_task(curr_state_np)
                analyzer = Analyzer(index=avail_idx)

                learner = maml.clone()

                em, _ = self.run_model(learner, analyzer, HORIZON * 2, data_index=avail_idx, early_finish=False)
                # Horizon for loop learner
                # Single adaptation gradient update 시작
                loss = self.model.conjugate_loss(memory=em)
                loss = loss / len(em.reward_list)

                learner.adapt(loss)
                # temp_model = type(self.model)(vocab_size=NUM_OF_CELLS, embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                #             output_size=NUM_OF_CELLS, learning_rate=LEARNING_RATE)
                # temp_model.to(DEVICE)
                # for (n1, x1), (n2, x2) in zip (self.model.named_parameters(), temp_model.named_parameters()):
                #     x2.data = x1.data.clone()
                # grad = torch.autograd.grad(loss, self.model.parameters(), create_graph= True)
                # print(grad)
                # for idx, (name, param) in enumerate(temp_model.named_parameters()):
                #     param.data = param.data - META_LEARNING_RATE * grad[idx]
                em, _ = self.run_model(learner, analyzer, HORIZON * 2, data_index=avail_idx, early_finish=False)
                step_loss = self.model.conjugate_loss(memory=em)

                if grad is None:
                    grad = torch.autograd.grad(step_loss, learner.parameters())
                else:
                    grad += torch.autograd.grad(step_loss, learner.parameters())
                # print(grad)
            # grad /= torch.FloatTensor(META_LEARNING_STEP).to(DEVICE)
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                param.grad = (grad[idx] / TASKS_PER_META_BATCH)
            optimizer.step()
            # opt.zero_grad()
            # step_loss.backward()
            # opt.step()
            #
            # em, _ = self.run_model(learner, analyzer, HORIZON * 2, data_index=avail_idx, early_finish=False)
            # meta_loss = self.model.conjugate_loss(memory=em)
            # opt.zero_grad()
            # meta_loss.backward()
            # opt.step()
            # grad = torch.autograd.grad(meta_loss, self.model.parameters())
            # for idx, (name, param) in enumerate(self.model.named_parameters()):
            #     param.grad = grad[idx]
            # self.model.optimizer.step()

    def run(self):
        o_num_list_per_episode = []
        value_list_per_episode = []
        optimizer = optim.Adam(lr=LEARNING_RATE, params= self.model.parameters())
        analyzer = Analyzer(self.data_index)
        self.model.avail_index = self.data_index.copy()
        self.model.done = True

        avail = np.zeros((NUM_OF_CELLS, 1))
        h_state = np.concatenate((avail, self.hilbert.getCoords(ORDER)), axis=1)
        z_state = np.concatenate((avail, self.z.getCoords(ORDER)), axis=1)
        h_num = analyzer.l2_norm_locality(h_state)
        z_num = analyzer.l2_norm_locality(z_state)

        observation = self.reset(self.model.state)
        self.init_o_num = analyzer.l2_norm_locality(observation)
        global_min_o_num = self.init_o_num  # 최소 locality 의 역(reverse) 값
        # global_min_state = np.empty(1)

        print(f'hilbert : {h_num} , Z : {z_num}, Initial ({self.initial_curve}) : {self.init_o_num}')

        for episode in range(self.MAX_EPISODE):  # 최대 에피소드 수만큼 반복
            em, info = self.run_model(self.model, analyzer, self.MAX_STEP * 2, self.data_index, early_finish=True)

            min_o_num = np.min(info)
            step = len(info)
            mean_o_num = np.mean(info)

            if global_min_o_num > min_o_num:
                global_min_o_num = min_o_num

            print(f'[GLOB {global_min_o_num:.3f} / INIT {self.init_o_num:.3f} / EP_NUM {episode}] '
                  f'stop after {step}.. so far [MIN:{min_o_num:.3f}/MEAN:{mean_o_num:.3f}/RW {np.mean(em.reward_list):.3f}]')
            o_num_list_per_episode = o_num_list_per_episode + info
            # with torch.no_grad():
            #     value_list_per_episode = value_list_per_episode + em.value()
            # Step For 문 종료
            self.model.update(em, optimizer)
            # em.empty()
            em.episode_reset()
        # Episode For 문 종료
        plt.plot(o_num_list_per_episode, 'o', markersize=2)
        plt.xlabel('step')
        plt.ylabel('Reverse of the locality')
        plt.axhline(y=self.init_o_num, color='r', linestyle='-')
        plt.axhline(y=global_min_o_num, color='g', linestyle='-')

        plt.plot(value_list_per_episode, 'o', markersize=2)
        plt.show()

        return global_min_o_num

    def run_model(self, model, analyzer, num_of_steps, data_index, early_finish):
        em = EpisodeMemory()
        # replay = ch.ExperienceReplay().to(DEVICE)
        action_list = []
        o_num_list = []

        if model.done:
            model.state = np.arange(0, NUM_OF_CELLS)
            model.his = (torch.zeros(1, 1, self.hidden_size).to(DEVICE), torch.zeros(1, 1, self.hidden_size).to(DEVICE))
            model.done = False
        else:
            model.his = (model.his[0].data.to(DEVICE), model.his[1].data.to(DEVICE))

        observation = self.reset(model.state)
        init_o_num = min_o_num = analyzer.l2_norm_locality(observation)
        curr_state_t = torch.from_numpy(model.state).long().to(DEVICE)

        for step in range(num_of_steps):
            value, logit, model.his = \
                model(input=curr_state_t, avail_id=data_index, his=model.his, action=action_list)
            prob = torch.softmax(logit, dim=-1)
            log_prob = torch.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum()
            m = Categorical(prob)
            a = m.sample().data
            action_list.append(a.item())
            # log_prob = log_prob.view(-1).gather(0, a)
            log_prob = m.log_prob(a)

            # Reward Part
            if len(action_list) <= 1:
                reward = 0
            else:
                model.state[action_list[0]], model.state[action_list[1]] \
                    = model.state[action_list[1]], model.state[action_list[0]]
                # Action 에 따른 Lc 값 측정
                observation = self.step(observation, action_list)
                o_num = analyzer.l2_norm_locality(observation)
                o_num_list.append(o_num)
                reward, model.done = self.give_reward(min_o_num, o_num, action_list, init_o_num, finish=early_finish)

                if reward > 0:
                    min_o_num = o_num

                action_list = []
            next_state_t = torch.from_numpy(model.state).long().to(DEVICE)
            em.append(value, log_prob, reward, entropy)
            # replay.append(curr_state_t, a, reward, next_state=next_state_t, done=model.done, log_prob= log_prob, value=value)
            curr_state_t = next_state_t

            if model.done:
                break

        # for step 종료
        return em, o_num_list

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

    def give_reward(self, min_num, num, action,init_o_num,  finish):
        done = False
        if min_num < num:
            if finish:
                done = True
            reward = - (num - min_num) ** 2
        elif min_num == num:
            if action[0] == action[1]:
                reward = -1
            else:
                reward = -0.01
        else:
            reward = (init_o_num - num) ** 2
        return reward, done


'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


class Analyzer():
    def __init__(self, index, init_state=None):
        self.scan_index = index

        avail = np.zeros((NUM_OF_CELLS, 1))
        avail[index] = 1
        if init_state is not None:
            self.init_state = np.concatenate((avail, init_state), axis=1)

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
    print(f'Avail data set {scan_index}')

    # if NOTEBOOK:
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     showPoints(sample_data, order=ORDER, dim=DIM, ax=ax, index=False)
    #
    #     if INIT_CURVE == 'hilbert':
    #         showlineByIndexorder(np.array(HilbertCurve(DIM).getCoords(ORDER)), INDEX_TO_COORDINATE, ax, index=False)
    #     elif INIT_CURVE == 'zig-zag':
    #         grid_index = np.arange(NUM_OF_CELLS)
    #         showlineByIndexorder(grid_index, INDEX_TO_COORDINATE, ax)
    #     plt.show(block=True)


    env = Env(data_index=scan_index, max_episode=MAX_EPISODE, max_step=MAX_STEP, init_curve=INIT_CURVE)
    if LOAD_AND_SAVE :
        print('load meta model (y / any key except y )?')
        answer = 'y'
        if answer =='y' :
            name = 'model_ORDER+' + str(ORDER) + '_DIM+' + str(DIM) + '.pt'
            try:
                saved_model = torch.load(name)
                env.model.load_state_dict(saved_model)
            except FileNotFoundError:
                print(f'Cannot found meta model(file name : {name})')
                print('start meta-learning...')
                env.meta_run()
                print(f'done ! save meta model (file name : {name})')
                torch.save(env.model.state_dict(), name)

    result_value = env.run()
    print(f'Recorded the minimum reverse of the locality :{result_value}')

    # if NOTEBOOK:
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     showPoints(sample_data, ax, index=False)
    #     showlineByIndexorder(result_state[:, 1:3].reshape([-1, 2]), ax, index=False)
    #     plt.show(block=True)

    # Test trained model
    if TEST :
        np.random.seed(175)

        print(f'Start testing trained model ... ')
        test_index = np.random.choice(2 ** (DIM * ORDER), size=DATA_SIZE, replace=False)
        print(f'Avail data set {test_index}')
        env.data_index = test_index
        result_value = env.run()

        print(f'[TEST]Recorded the minimum reverse of the locality :{result_value}')

    sys.exit(0)