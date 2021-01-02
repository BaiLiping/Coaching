from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

'''
thigh_actuator_kp=[0.42916187,-1.35645348,-2.30216236,-0.61791816,-0.91821599]
thigh_actuator_kd=[0.12490907,-2.28855392,0.76335732, 0.23505598,-0.39615837]
leg_actuator_kp=[0.12228319,-0.42720977,-0.47019598,-0.11091503,-0.20794748]
leg_actuator_kd=[0.02967504,-0.65914378,0.21477365,-0.09003069,-0.06619213]
foot_actuator_kp=[-0.06219511, -1.16343272,  0.89993129,  0.44874288, -0.97117291]
foot_actuator_kd=[0.11365922,-0.3601859,-0.03225303,-0.05621569,-0.52937106]
'''
thigh_actuator_kp=[-2,-2,-0.5,-1]
thigh_actuator_kd=[-2,1, 0.2,-0.4]
leg_actuator_kp=[-0.4,-0.5,-0.1,-0.2]
leg_actuator_kd=[-1,0.2,-1,-0.1]
foot_actuator_kp=[-2, 1, 0.5, -1]
foot_actuator_kd=[-0.4,-0.1,-0.1,-0.5]

environment = gym.make('Hopper-v3')
print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
while not terminal:

	rooty=states[1]
	velocity_rooty=states[7]

	thigh_angle=states[2]
	thigh_angular_velocity=states[8]

	leg_angle=states[3]
	leg_angular_velocity=states[9]

	foot_angle=states[4]
	foot_angular_velocity=states[10]

	thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty+thigh_actuator_kp[1]*thigh_angle+thigh_actuator_kd[1]*thigh_angular_velocity+thigh_actuator_kp[2]*leg_angle+thigh_actuator_kd[2]*leg_angular_velocity+thigh_actuator_kp[3]*foot_angle+thigh_actuator_kd[3]*foot_angular_velocity
	leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty+leg_actuator_kp[1]*thigh_angle+leg_actuator_kd[1]*thigh_angular_velocity+leg_actuator_kp[2]*leg_angle+leg_actuator_kd[2]*leg_angular_velocity+leg_actuator_kp[3]*foot_angle+leg_actuator_kd[3]*foot_angular_velocity
	foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty+foot_actuator_kp[1]*thigh_angle+foot_actuator_kd[1]*thigh_angular_velocity+foot_actuator_kp[2]*leg_angle+foot_actuator_kd[2]*leg_angular_velocity+foot_actuator_kp[3]*foot_angle+foot_actuator_kd[3]*foot_angular_velocity
	actions=[thigh_actions,leg_actions,foot_actions]

	states, reward, terminal,info = environment.step(actions)
	print('y_position %s y_velocity %s' %(states[1],states[7]))
	episode_reward+=reward
print(episode_reward)
