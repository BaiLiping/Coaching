from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym



kp=30
ki=0.001
kd=2.26


angle_record=[]
velocity_record=[]
actions_record=[]
environment = gym.make('InvertedPendulum-v2')

print('testing PID controller')
episode_reward=0
states = environment.reset()
terminal=False
integral=0
while not terminal:
    #environment.render()
    integral+=states[1]
    actions = kp*states[1]+ki*integral+kd*states[3]
        # disturbances
        # if step_count%100==0:
        # for p in range(2):
        #     actions=-1
        #     states,rewar,terminal,info=environment.step(actions)

        # for q in range(2):
        #     actions=1
        #     states,rewar,terminal,info=environment.step(actions)
    states, reward, terminal,info = environment.step(actions)
    angle_record.append(states[1])
    velocity_record.append(states[3])
    actions_record.append(actions)
    episode_reward+=reward
print(episode_reward)


length=len(angle_record)
x=range(length)
plt.figure(figsize=(13,8))
plt.plot(x,angle_record,label='Angle',color='black')
plt.plot(x,velocity_record,label='Angular Velocity',color='blue',alpha=0.5)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#ax1.set_ylim([-0.1, 0.1])
plt.ylim(-0.15,0.15)
plt.yticks(fontsize=25)
plt.xlabel('Steps', fontsize=30)
plt.xticks(fontsize=25)
plt.savefig('ip_PID.png')
plt.close()

plt.figure(figsize=(13,8))
plt.plot(x,actions_record,label='PID Actions',color='royalblue',alpha=0.5)
#ax2.xlabel('Steps', fontsize='large')
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#.set_ylim([-.25,.25])
plt.ylim(-0.25,0.25)
plt.yticks(fontsize=25)
plt.xlabel('Steps', fontsize=30)
plt.xticks(fontsize=25)
plt.savefig('ip_PID_actions.png')