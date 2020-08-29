import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch.autograd import Variable
from multiprocessing import Process, Pipe

'''
    Actor-Critic CartPole Version
'''

try:
    xrange = xrange
except:
    xrange = range

NOTEBOOK = False
TEST = False
CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_STEP = 200
CAPACITY = 10000
NUM_ADVANCED_STEP = 5  # 총 보상을 계산할 때 Advantage 학습을 할 단계 수

import gym

# 상수 정의
ENV = 'CartPole-v0'  # 태스크 이름

# -------- Hyper Parameter --------------- #
LEARNING_RATE = 1e-2  # 학습률
GAMMA = 0.99  # 시간 할인율
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 0.5
OFFSET = 0  # 기존 state 좌표 값 외에 신경망에 추가로 들어갈 정보의 갯수
NUM_PROCESSES = 32  # 동시 실행 환경 수



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
    return discounted_r


class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''

    def __init__(self, num_steps, num_processes, obs_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_size).to(DEVICE)
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


class SFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SFCNet, self).__init__()
        self.hidden_size = hidden_size or ((input_size + output_size) // 2)

        self.input_nn = nn.Linear(input_size, self.hidden_size)
        self.hidden_nn = nn.Linear(self.hidden_size, self.hidden_size)
        self.actor_nn = nn.Linear(self.hidden_size, output_size)
        self.critic_nn = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        output = torch.relu(self.input_nn(input))
        output = torch.relu(self.hidden_nn(output))
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
        actor_output, values = self.model(rollouts.observations[:-1].view(-1, self.num_states))
        log_probs = torch.log_softmax(actor_output, dim= 1)

        action_log_probs = log_probs.gather(1, rollouts.actions.view(-1,1))

        probs = F.softmax(actor_output, dim=1)

        # 엔트로피 H : action이 확률적으로 얼마나 퍼져 있는가? (비슷한 확률의 다중 액션 -> high, 단일 액션 -> low)
        entropy = -( (log_probs * probs)).sum(-1).mean()

        values = values.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
        action_log_probs = action_log_probs.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)

        # advantage(행동가치(할인 총 보상, discounted reward)-상태가치(critic value)) 계산
        advantages = rollouts.returns[:-1] - values

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        action_gain = ( (action_log_probs) * advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (value_loss * VALUE_COEFF - action_gain - entropy * ENTROPY_COEFF)
        self.model.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록, 경사를 0.5 이하로 제한함(클리핑)

        self.optimizer.step()  # 결합 가중치 수정

    def decide_action(self, state):
        with torch.no_grad():
            action, _ = self.model(state)
            a = torch.softmax(action, dim= 1)
            a = a.multinomial(num_samples=1)

            # equivalent with ...
            # a, b = np.random.choice(dist, size=2, replace=False, p=dist)
            # a = np.argmax(dist == a)
            # b = np.argmax(dist == b)
            return a

    def compute_value(self, state):
        _, value = self.model(state)
        return value


class Agent():
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions, 32)

    def update_policy_function(self, history):
        self.brain.update(history)

    def get_action(self, state):
        action = self.brain.decide_action(state)
        return action

    def get_value(self, state):
        value = self.brain.compute_value(state)
        return value


class Env():
    def __init__(self, max_episode):
        self.envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]
        # 모든 에이전트가 공유하는 Brain 객체를 생성
        self.MAX_EPISODE = max_episode
        self.num_action_space = self.envs[0].action_space.n
        self.num_observation_space = self.envs[0].observation_space.shape[0]

        # Reward 설정용
        self.agent = Agent(self.num_observation_space, self.num_action_space)


    '''
    Agent로 부터 행동을 선택하고 그 행동에 맞춰서 이후 상태 관찰,
    관찰에 기반하여 보상을 계산하고 이들을 버퍼에 저장
    버퍼에 충분히 Transition을 저장했다고 생각하면 신경망(Q 함수) 업데이트
    '''

    def run(self):

        obs_shape = self.num_observation_space
        current_obs = torch.zeros(NUM_PROCESSES, self.num_observation_space)  # torch.Size([16, 4])
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rollouts 객체
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 마지막 에피소드의 보상
        obs_np = np.zeros([NUM_PROCESSES, self.num_observation_space])  # Numpy 배열
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
        each_step = np.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록
        episode = 0  # 환경 0의 에피소드 수

        # 초기 상태로부터 시작
        obs = [self.envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 가장 최근의 obs를 저장

        # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
        rollouts.observations[0].copy_(current_obs)

        for j in range(self.MAX_EPISODE * NUM_PROCESSES):  # 최대 에피소드 수만큼 반복
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action = self.agent.get_action(rollouts.observations[step])

                actions = action.squeeze(1).cpu().numpy()

                # 한 단계를 실행
                for i in range(NUM_PROCESSES):
                    print(f' {i} : {actions[i]}')
                    obs_np[i], reward_np[i], done_np[i], _ = self.envs[i].step(actions[i])

                    if done_np[i]:  # 단계 수가 200을 넘거나, 봉이 일정 각도 이상 기울면 done이 True가 됨

                        # 환경 0일 경우에만 출력
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i] + 1))
                            episode += 1

                        # 보상 부여
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 도중에 봉이 넘어지면 페널티로 보상 -1 부여
                        else:
                            reward_np[i] = 1.0  # 봉이 쓰러지지 않고 끝나면 보상 1 부여

                        each_step[i] = 0  # 단계 수 초기화
                        obs_np[i] = self.envs[i].reset()  # 실행 환경 초기화

                    else:
                        reward_np[i] = 0.0  # 그 외의 경우는 보상 0 부여
                        each_step[i] += 1
                print('----------------------------')
                # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌

                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])

                # 마지막 에피소드의 총 보상을 업데이트
                final_rewards *= masks  # done이 false이면 1을 곱하고, true이면 0을 곱해 초기화
                # done이 false이면 0을 더하고, true이면 episode_rewards를 더해줌
                final_rewards += (1 - masks) * episode_rewards

                # 에피소드의 총보상을 업데이트
                episode_rewards *= masks  # done이 false인 에피소드의 mask는 1이므로 그대로, true이면 0이 됨

                # 현재 done이 true이면 모두 0으로
                current_obs *= masks

                # current_obs를 업데이트
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 최신 상태의 obs를 저장

                rollouts.insert(current_obs, action.data, reward, masks)

                # advanced 학습 for문 끝
            # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산
            with torch.no_grad():
                next_value = self.agent.get_value(rollouts.observations[-1]).detach()
            rollouts.compute_returns(next_value)

            # 신경망 및 rollout 업데이트
            self.agent.update_policy_function(rollouts)
            rollouts.after_update()

            # 환경 갯수를 넘어서는 횟수로 200단계를 버텨내면 성공
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('연속성공')
                break

'''
주어진 state와 활성화된 데이터를 기반으로 reward를 위한 metrics을 측정하는 함수
'''


'''
index (n) 은 다음과 같이 좌표로 표시됨
n 의 최댓값은 DIM * ORDER - 1 
좌표 값은 ( n // (DIM * ORDER), n % (DIM * ORDER) ) 
'''


def main():
    env = Env(max_episode=3000)
    env.run()



if __name__ == '__main__':
    main()