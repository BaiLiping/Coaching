from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

tuning_episode=1000
environment = gym.make('InvertedPendulum-v2')
print('tuning PID controller')
for i in tqdm(range(tuning_episode)):
	episode_reward=0
	states = environment.reset()
	terminal=False
	while not terminal:
		angle=states[1]
		angular_velocity=states[3]
		print('angle: %s velocity: %s' %(angle,angular_velocity))
		actions = input("Please enter the action:\n")
		states, reward, terminal,info = environment.step(actions)
		episode_reward+=reward
	print(episode_reward)
