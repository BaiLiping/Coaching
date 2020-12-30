from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym


thigh_actuator_kp=[0.42916187,-1.35645348,-2.30216236,-0.61791816,-0.91821599]
thigh_actuator_kd=[0.12490907,-2.28855392,0.76335732, 0.23505598,-0.39615837]
leg_actuator_kp=[0.12228319,-0.42720977,-0.47019598,-0.11091503,-0.20794748]
leg_actuator_kd=[0.02967504,-0.65914378,0.21477365,-0.09003069,-0.06619213]
foot_actuator_kp=[-0.06219511, -1.16343272,  0.89993129,  0.44874288, -0.97117291]
foot_actuator_kd=[0.11365922,-0.3601859,-0.03225303,-0.05621569,-0.52937106]



environment = gym.make('Hopper-v3')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
while not terminal:
	rootz=states[0]
	velocity_rootz=states[6]

	rooty=states[1]
	velocity_rooty=states[7]

	thigh_angle=states[2]
	thigh_angular_velocity=states[8]

	leg_angle=states[3]
	leg_angular_velocity=states[9]

	foot_angle=states[4]
	foot_angular_velocity=states[10]


    actions = kp*states[1]+kd*states[3]
    states, reward, terminal,info = environment.step(actions)
    print('states: ',states)
    print('actions: ',actions)
    episode_reward+=reward
print(episode_reward)
