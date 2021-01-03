from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym


kp=15
kd=2.25

record=[]
y_record=[]
environment = gym.make('InvertedPendulum-v2')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
while not terminal:
    actions = kp*states[1]+kd*states[3]
    states, reward, terminal,info = environment.step(actions)
    print('velocity: ',states[3])
    record.append(states[3])
    y_record.append(states[1])
    print('actions: ',actions)
    episode_reward+=reward
print(episode_reward)

x=range(len(record))
plt.plot(x,record)
plt.show()
plt.plot(x,y_record)
plt.show()