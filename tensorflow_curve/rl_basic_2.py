import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts, trajectory

tf.compat.v1.enable_v2_behavior()

# @title 기본 제목 텍스트
class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 0:
            self._episode_ended = True
        elif action == 1:
            new_card = np.random.randint(1, 3)
            self._state += new_card
        elif action == 2:
            new_card = np.random.randint(3, 6)
            self._state += new_card
        elif action == 3:
            new_card = np.random.randint(7, 9)
            self._state += new_card
        elif action == 4:
            new_card = np.random.randint(10, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 ~ 4.')

        # 에피소드가 끝나면 reward 부여 후 종료
        if self._episode_ended or self._state >= 21:
            if self._state < 21:
                reward = (self._state - 21)
            elif self._state == 21:
                reward = 1
            else:
                reward = -21

            self._episode_ended = True
            termination_info = ts.termination(
                np.array([self._state], dtype=np.int32), reward)
            return termination_info
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

get_new_card_action = np.random.randint(1, 4)
end_round_action = 0

environment = CardGameEnv()
time_step = environment.reset()
print('초기화 후 time_step: ', time_step)
cumulative_reward = time_step.reward

while not time_step.is_last() :
    time_step = environment.step(get_new_card_action)
    print('action : ', get_new_card_action, time_step)
    cumulative_reward += time_step.reward

print('Final Reward = ', cumulative_reward)

# 위에서 만든 카드 게임 Environment
# env_name = 'CartPole-v0'
# card_env = suite_gym.load(env_name)

card_env = CardGameEnv()
tf_card_env = tf_py_environment.TFPyEnvironment(card_env)

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

# optimizer, net 정의
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
tf_env_obs_spec = tf_card_env.observation_spec()
tf_env_ts_spec = tf_card_env.time_step_spec()
tf_env_act_spec = tf_card_env.action_spec()

# 훈련 몇 회 했는지 count 할 때 사용
train_step_counter = tf.compat.v2.Variable(0)

card_actor_net = actor_distribution_network.ActorDistributionNetwork(
    input_tensor_spec=tf_env_obs_spec,
    output_tensor_spec=tf_env_act_spec,
    activation_fn=tf.nn.relu)

# tf_env_ts_spec.observation == tf_env_ts_spec.observation : True
card_value_net = value_network.ValueNetwork(
    input_tensor_spec=tf_env_ts_spec.observation)

tf_card_agent = ppo_agent.PPOAgent(
    time_step_spec=tf_env_ts_spec,
    action_spec=tf_env_act_spec,
    optimizer=optimizer,
    actor_net=card_actor_net,
    value_net=card_value_net,
    train_step_counter=train_step_counter)

tf_card_agent.initialize()

from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer_max_length = 100000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_card_agent.collect_data_spec,
    batch_size=tf_card_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)

from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver

num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
avg_returns = tf_metrics.AverageReturnMetric()

observers = [replay_buffer.add_batch, num_episodes, env_steps, avg_returns]

driver = dynamic_episode_driver.DynamicEpisodeDriver(
    tf_card_env, tf_card_agent.collect_policy, observers, num_episodes=1)

for i in range(1000):
    final_time_step = driver.run()
    experiences = replay_buffer.gather_all()
    train_loss = tf_card_agent.train(experiences)
    print(num_episodes.result().numpy())
    print(experiences)
    print('\n')
    replay_buffer.clear()
