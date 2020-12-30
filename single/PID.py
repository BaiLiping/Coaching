from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

#kp=1.77327564e+00
#kd=-2.60674054e-02
kp=25
kd=2.27645649
#kp=3.02458142 
#kd=2.50691026
#kp=1.67000663 
#kd=-2.27637994
# Pre-defined or custom environment
environment = gym.make('InvertedPendulum-v2')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
while not terminal:
    actions = kp*states[1]+kd*states[3]
    states, reward, terminal,info = environment.step(actions)
    print('states: ',states)
    print('actions: ',actions)
    episode_reward+=reward
print(episode_reward)
