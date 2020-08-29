from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import random_py_environment
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network, actor_distribution_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common as common_utils, common
from tf_agents.utils import nest_utils
from tf_agents.utils import eager_utils
from tf_agents.environments import suite_gym
from tf_agents.utils import value_ops

import copy

tf.compat.v1.enable_v2_behavior()

env_name = "CartPole-v0"  # @param {type:"string"}
num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

env = suite_gym.load(env_name)

print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)
tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

# Policies
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


# Metrics and Evaluation
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

# Replay Buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=eval_env.batch_size,
    max_length=replay_buffer_capacity)


# Data Collection
def collect_episode(environment, policy, num_episodes):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

num_iterations = 1

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    print('train 전 : ', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))
    experience = replay_buffer.gather_all()
    # 일반적인 훈련 방법 train 함수 호출
    train_loss = tf_agent.train(experience)
    print('train 후 : ', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))

    ###### 직접 total_loss 를 만들어서 학습하는 방법 (train 함수 코드 참조) ########
    print('total_loss 만들기 전 : ', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))
    non_last_mask = tf.cast(tf.math.not_equal(experience.next_step_type, ts.StepType.LAST), tf.float32)
    discounts = non_last_mask * experience.discount * tf_agent._gamma
    returns = value_ops.discounted_return(experience.reward, discounts, time_major=False)
    time_step = ts.TimeStep(experience.step_type, tf.zeros_like(experience.reward), tf.zeros_like(experience.discount),
                            experience.observation)
    with tf.GradientTape() as tape:
        loss_info = tf_agent.total_loss(time_step, experience.action, tf.stop_gradient(returns), weights=None)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')

    variables_to_train = tf_agent._actor_network.trainable_weights
    if tf_agent._baseline:
        variables_to_train += tf_agent._value_network.trainable_weights
    grads = tape.gradient(loss_info.loss, variables_to_train)
    print('total_loss 에 대한 grads : ', tf.reduce_mean(grads[0], axis=1))

    grads_and_vars = list(zip(grads, variables_to_train))
    if tf_agent._gradient_clipping:
        grads_and_vars = eager_utils.clip_gradient_norms(
            grads_and_vars, tf_agent._gradient_clipping)

    if tf_agent._summarize_grads_and_vars:
        eager_utils.add_variables_summaries(grads_and_vars, tf_agent.train_step_counter)
        eager_utils.add_gradients_summaries(grads_and_vars, tf_agent.train_step_counter)

    tf_agent._optimizer.apply_gradients(grads_and_vars, global_step=tf_agent.train_step_counter)
    print('optimizer 로 loss 학습 후 :', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))

    ####### 직접 loss 만들어서 훈련하기 종료 #########################
    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

u = copy.deepcopy(actor_net)


# print(u.get_weights())

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


with tf.GradientTape() as test_tape:
    with tf.GradientTape() as train_tape:
        curr_loss = loss(actor_net.submodules[4](np.random.random([100, 100])), tf.random.uniform([100, 2]))
    dW, db = train_tape.gradient(curr_loss, actor_net.submodules[4].weights)

    z = actor_net.copy()
    z.create_variables()
    z.set_weights(actor_net.get_weights())

    z.submodules[4].kernel = tf.subtract(actor_net.submodules[4].kernel, tf.multiply(0.003, dW))
    z.submodules[4].bias = tf.subtract(actor_net.submodules[4].bias, tf.multiply(0.003, db))

    print(dW, db)
    curr_loss = loss(z.submodules[4](np.random.random([100, 100])), tf.random.uniform([100, 2]))
dW, db = test_tape.gradient(curr_loss, actor_net.submodules[4].trainable_variables)

print(dW, db)

### 복사한 network 를 새로운 agent 에 이식한 후 진행되는 meta-training ################


for _ in range(num_iterations):
    with tf.GradientTape() as test_tape:
        with tf.GradientTape() as train_tape:
            # Collect a few episodes using collect_policy and save to the replay buffer.
            collect_episode(eval_env, tf_agent.collect_policy, collect_episodes_per_iteration)

            # Use data from the buffer and update the agent's network.
            experience = replay_buffer.gather_all()

            # Compute loss and grad for the original Network
            # Referred the _train function of the reinforce algorithm
            # https://github.com/tensorflow/agents/blob/master/tf_agents/agents/reinforce/reinforce_agent.py
            non_last_mask = tf.cast(tf.math.not_equal(experience.next_step_type, ts.StepType.LAST), tf.float32)
            discounts = non_last_mask * experience.discount * tf_agent._gamma
            returns = value_ops.discounted_return(experience.reward, discounts, time_major=False)
            time_step = ts.TimeStep(experience.step_type, tf.zeros_like(experience.reward),
                                    tf.zeros_like(experience.discount),
                                    experience.observation)

            loss_info = tf_agent.total_loss(time_step, experience.action, tf.stop_gradient(returns), weights=None)
            tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
            print('loss : ', loss_info)
        variables_to_compute_grad = tf_agent._actor_network.trainable_weights
        if tf_agent._baseline:
            variables_to_compute_grad += tf_agent._value_network.trainable_weights
        grads = train_tape.gradient(loss_info.loss, variables_to_compute_grad)

        # Now make new agent but uses copied actor_net network
        copied_tf_agent = reinforce_agent.ReinforceAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            actor_network=tf_agent._actor_network.copy(),
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
            normalize_returns=True,
            train_step_counter=tf.compat.v2.Variable(0))
        copied_tf_agent.initialize()
        copied_tf_agent._actor_network.set_weights(tf_agent._actor_network.get_weights())

        variables_to_train = copied_tf_agent._actor_network.trainable_weights
        if copied_tf_agent._baseline:
            variables_to_train += copied_tf_agent._value_network.trainable_weights

        print('학습 전(original) : ', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))
        print('학습 전(copy) : ', tf.reduce_mean(copied_tf_agent.trainable_variables[1], axis=1))

        for i in range(len(variables_to_train)):
            weight = copied_tf_agent._actor_network.trainable_weights[i]
            weight.assign_sub(0.3 * grads[i])
        print('학습 후(original) : ', tf.reduce_mean(tf_agent.trainable_variables[1], axis=1))
        print('학습 후(copy) : ', tf.reduce_mean(copied_tf_agent.trainable_variables[1], axis=1))

        replay_buffer.clear()

        # Compute loss and grad for the cloned Network with respect to original Network
        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(train_env, copied_tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()

        # Copied Network 에 대해서 loss 생성 후 grad 계산
        non_last_mask = tf.cast(tf.math.not_equal(experience.next_step_type, ts.StepType.LAST), tf.float32)
        discounts = non_last_mask * experience.discount * copied_tf_agent._gamma
        returns = value_ops.discounted_return(experience.reward, discounts, time_major=False)
        time_step = ts.TimeStep(experience.step_type, tf.zeros_like(experience.reward),
                                tf.zeros_like(experience.discount),
                                experience.observation)
        meta_loss_info = copied_tf_agent.total_loss(time_step, experience.action, tf.stop_gradient(returns), weights=None)
        tf.debugging.check_numerics(meta_loss_info.loss, 'Loss is inf or nan')
        print('meta loss : ', meta_loss_info)
    variables_to_train = tf_agent._actor_network.trainable_weights
    if tf_agent._baseline:
        variables_to_train += tf_agent._value_network.trainable_weights
    grads = test_tape.gradient(meta_loss_info.loss, variables_to_train)
    print('meta grad ! :', tf.reduce_mean(grads[0], axis=1))
