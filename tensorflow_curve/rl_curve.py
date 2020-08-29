import gym
import gym_curve
import unittest
from cherry import envs
import cherry as ch

SEED = 42

def test_env():
    env = 'curve-v0'
    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Logger(env)
    env = envs.Runner(env)
    replay = ch.ExperienceReplay()


    ob = env.reset()
    print(ob)
    env.step(0)

test_env()
