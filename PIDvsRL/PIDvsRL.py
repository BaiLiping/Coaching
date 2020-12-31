from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

test_episodes=10

ip_pid_episode_record=[]
ip_rl_episode_record=[]
ip_rl = Agent.load(directory='Inverted_Pendulum_RL', format='numpy')
internals = ip_rl.initial_internals()
for i in range(test_episodes):
	kp=25
	kd=2.27645649
	environment = gym.make('InvertedPendulum-v2')

	episode_reward=0
	states = environment.reset()
	terminal=False
	while not terminal:
		actions = kp*states[1]+kd*states[3]
		states, reward, terminal,info = environment.step(actions)
		episode_reward+=reward
	ip_pid_episode_record.append(episode_reward)

	environment_rl = Environment.create(environment='gym', level='InvertedPendulum-v2')
	episode_reward=0
	states = environment_rl.reset()
	terminal=False
	while not terminal:
		actions, internals = ip_rl.act(states=states, internals=internals, independent=True, deterministic=True)
		states, terminal, reward = environment_rl.execute(actions=actions)
		episode_reward+=reward
	ip_rl_episode_record.append(episode_reward)
ip_rl.close()
environment.close()
environment_rl.close()

double_pid_episode_record=[]
double_rl_episode_record=[]
double_rl = Agent.load(directory='Double_RL', format='numpy')
internals = double_rl.initial_internals()
for i in range(test_episodes):
	kp=[-0.54124971, -3.05534616]
	kd=[-0.47012709, -0.70023993]
	environment_control = gym.make('InvertedDoublePendulum-v2')
	episode_reward=0
	states = environment_control.reset()
	terminal=False
	theta_states=[]
	while not terminal:
		actions_predict=kp[0]*states[0]+kp[1]*states[2]+kd[0]*states[6]+kd[1]*states[7]
		states, reward, terminal,info = environment_control.step(actions_predict)
		episode_reward+=reward
	double_pid_episode_record.append(episode_reward)

	environment_rl = Environment.create(environment='gym', level='InvertedDoublePendulum-v2')
	episode_reward=0
	states = environment_rl.reset()
	terminal=False
	while not terminal:
		actions, internals = double_rl.act(states=states, internals=internals, independent=True, deterministic=True)
		states, terminal, reward = environment_rl.execute(actions=actions)
		episode_reward+=reward
	double_rl_episode_record.append(episode_reward)
double_rl.close()
environment_control.close()
environment_rl.close()

hopper_pid_episode_record=[]
hopper_rl_episode_record=[]
hopper_rl = Agent.load(directory='Hopper_RL', format='numpy')
internals = hopper_rl.initial_internals()
for i in range(test_episodes):
	thigh_actuator_kp=[-2,-2,-0.5,-1]
	thigh_actuator_kd=[-1.6,1, 0.2,-0.4]
	leg_actuator_kp=[-0.4,-0.5,-0.1,-0.2]
	leg_actuator_kd=[-1,0.2,-1,-0.1]
	foot_actuator_kp=[-2, 1, 0.5, -1]
	foot_actuator_kd=[-0.4,-0.1,-0.1,-0.5]
	environment = gym.make('Hopper-v3')
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
		episode_reward+=reward
	hopper_pid_episode_record.append(episode_reward)

	environment_rl = Environment.create(environment='gym', level='Hopper-v3')
	episode_reward=0
	states = environment_rl.reset()
	terminal=False
	while not terminal:
		actions, internals = hopper_rl.act(states=states, internals=internals, independent=True, deterministic=True)
		states, terminal, reward = environment_rl.execute(actions=actions)
		episode_reward+=reward
	hopper_rl_episode_record.append(episode_reward)
hopper_rl.close()
environment.close()
environment_rl.close()

walker_pid_episode_record=[]
walker_rl_episode_record=[]
walker_rl = Agent.load(directory='Walker_RL', format='numpy')
internals = walker_rl.initial_internals()
for i in range(test_episodes):
	thigh_actuator_kp=[-3,-2]
	thigh_actuator_kd=[-2,1]
	leg_actuator_kp=[-.5,-0.5]
	leg_actuator_kd=[-.5,0.2]
	foot_actuator_kp=[-.5, 1]
	foot_actuator_kd=[-.5,-0.1]
	left_thigh_actuator_kp=[-3,-2]
	left_thigh_actuator_kd=[-2,1]
	left_leg_actuator_kp=[-.5,-0.5]
	left_leg_actuator_kd=[-.5,0.2]
	left_foot_actuator_kp=[-0.5, 1]
	left_foot_actuator_kd=[-0.5,-0.1]
	
	environment = gym.make('Walker2d-v3')
	episode_reward=0
	states = environment.reset()
	terminal=False
	while not terminal:

		rooty=states[1]
		velocity_rooty=states[10]

		thigh_angle=states[2]
		thigh_angular_velocity=states[11]

		leg_angle=states[3]
		leg_angular_velocity=states[12]

		foot_angle=states[4]
		foot_angular_velocity=states[13]

		left_thigh_angle=states[5]
		left_thigh_angular_velocity=states[14]

		left_leg_angle=states[6]
		left_leg_angular_velocity=states[15]

		left_foot_angle=states[7]
		left_foot_angular_velocity=states[16]

		thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty
		leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty
		foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty
		left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty
		left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty
		left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty
		actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]
		states, reward, terminal,info = environment.step(actions)
		episode_reward+=reward
	walker_pid_episode_record.append(episode_reward)

	environment_rl = Environment.create(environment='gym', level='Walker2d-v3')
	episode_reward=0
	states = environment_rl.reset()
	terminal=False
	while not terminal:
		actions, internals = walker_rl.act(states=states, internals=internals, independent=True, deterministic=True)
		states, terminal, reward = environment_rl.execute(actions=actions)
		episode_reward+=reward
	walker_rl_episode_record.append(episode_reward)

walker_rl.close()
environment.close()
environment_rl.close()


pid_record=[]
rl_record=[]

pid_record.append(int(np.sum(ip_pid_episode_record)/test_episodes))
pid_record.append(int(np.sum(double_pid_episode_record)/test_episodes*0.1))
pid_record.append(int(np.sum(hopper_pid_episode_record)/test_episodes))
pid_record.append(int(np.sum(walker_pid_episode_record)/test_episodes))

rl_record.append(int(np.sum(ip_rl_episode_record)/test_episodes))
rl_record.append(int(np.sum(double_rl_episode_record)/test_episodes*0.1))
rl_record.append(int(np.sum(hopper_rl_episode_record)/test_episodes))
rl_record.append(int(np.sum(walker_rl_episode_record)/test_episodes))





# data to plot

labels = ['Inverted\nPendulum', 'Double\nInverted\nPendulum', 'Hopper','Walker']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pid_record, width, label='PID')
rects2 = ax.bar(x + width/2, rl_record, width, label='RL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Score Over 10 Trials')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('PIDvsRL.png')
