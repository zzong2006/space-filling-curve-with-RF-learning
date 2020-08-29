import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)

total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_arms) #Set scoreboard for bandit arms to 0.

def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1

weights = torch.ones(num_arms, requires_grad= True)
soft_m = nn.Softmax(dim=-1)
output = Variable(soft_m(weights))

lr = 1e-3
i = 0
while i < total_episodes:
    # Choose action according to Boltzmann distribution.
    actions = soft_m(weights)
    dummyActs = actions.detach().numpy()
    a = np.random.choice(dummyActs, p=dummyActs)
    action = np.argmax(dummyActs == a)

    # Get our reward from picking one of the bandit arms.
    reward = pullBandit(bandit_arms[action])
    responsible_output = actions[action]
    loss = (-torch.log(responsible_output) * reward)

    # Update the network
    loss.backward()
    with torch.no_grad():
        weights -= weights.grad * lr
        weights.grad.zero_()

    total_reward[action] += reward

    if i % 50 == 0:
        print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
        # print("Weight : " + str(weights))
    i += 1
ww = weights.detach().numpy()
print("\nThe agent thinks arm " + str(np.argmax(ww).item() + 1) + " is the most promising....")
if np.argmax(ww).item() == np.argmax(-np.array(bandit_arms)):
    print("...and it was right!")
else:
    print("...and it was wrong!")