from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from gym.wrappers import Monitor

#environment = Environment.create(environment='gym', level='InvertedPendulum-v2')
environment=gym.make('InvertedPendulum-v2')
RL= Agent.load(directory='model5', format='numpy')
internals = RL.initial_internals()
actions_record=[]
theta_states=[]
for k in range(1):
    states = environment.reset()
    terminal=False
    integrals=0
    while not terminal:
        #environment.render()
        integrals+=states[1]
        temp=[states[1],integrals,states[3]]
        theta_states.append(temp)
        actions, internals = RL.act(states=states, internals=internals, independent=True, deterministic=True)
        #states, terminal, reward = environment.execute(actions=actions)
        states,reward,terminal,info=environment.step(actions)
        actions_record.append(actions)

theta=[row[0] for row in theta_states]
theta_velocity=[row[2] for row in theta_states]
summation=[row[1] for row in theta_states]

length=len(theta)
x=range(len(theta))
plt.figure(figsize=(13,9))
plt.plot(x,theta,label='Angle',color='black')
plt.plot(x,theta_velocity,label='Angular Velocity',color='blue',alpha=0.5)
plt.xlabel('Steps', fontsize=30)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#plt.ylim(-0.1,0.1)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.savefig('ip_RL.png')
plt.close()
plt.figure(figsize=(13,9))
plt.plot(x,actions_record,label='RL Actions',color='royalblue', alpha=0.5)
plt.xlabel('Steps', fontsize=30)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#plt.ylim(-0.1,0.1)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('ip_RL_actions.png')

RL.close()
environment.close()