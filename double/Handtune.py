from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

tuning_episode=1000
environment = gym.make('InvertedDoublePendulum-v2')
print('tuning PID controller')
for i in tqdm(range(tuning_episode)):
	episode_reward=0
	states = environment.reset()
	terminal=False
	while not terminal:
		sintheta1=states[1]
		velocity_theta1=states[6]
		print('sintheta1: %s velocity: %s' %(sintheta1,velocity_theta1))
		actions = input("Please enter the action:\n")
		states, reward, terminal,info = environment.step(actions)
		episode_reward+=reward
	print(episode_reward)
