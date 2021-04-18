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

environment=gym.make('InvertedDoublePendulum-v2')
states=environment.reset()
RL= Agent.load(directory='Double_Model', format='numpy')
internals = RL.initial_internals()
actions_record=[]

theta1_record=[]
theta2_record=[]

theta1_velocity_record=[]
theta2_velocity_record=[]

theta1_integral_record=[]
theta2_integral_record=[]

#states = environment.reset()
terminal=False
theta1_integral=0
theta2_integral=0

while not terminal:
    #environment.render()
    sintheta1=states[1]
    print('theta1:',sintheta1)
    theta1_record.append(sintheta1)
    theta1_integral+=sintheta1
    theta1_integral_record.append(theta1_integral)
    sintheta2=states[2]
    theta2_record.append(sintheta2)
    theta2_integral+=sintheta2
    theta2_integral_record.append(theta2_integral)
    theta1_velocity_record.append(states[6])
    print('velocity theta1:', states[6])
    theta2_velocity_record.append(states[7])
    actions, internals = RL.act(states=states, internals=internals, independent=True, deterministic=True)
    #states, terminal, reward = environment.execute(actions=actions)
    states,reward,terminal,info=environment.step(actions)
    actions_record.append(actions)

length=len(theta1_record)
x=range(length)
plt.figure(figsize=(13,8.5))
plt.plot(x,theta1_record,label='Lower Angle',color='black')
plt.plot(x,theta1_velocity_record,label='Lower Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta1_integral_record,label='Integrals',color='magenta')

plt.xlabel('Steps', fontsize=30)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#plt.ylim(-0.1,0.1)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.savefig('double_RL.png')
#plt.show()
plt.close()
plt.figure(figsize=(13,8.5))

plt.plot(x,theta2_record,label='Upper Angle',color='black')
plt.plot(x,theta2_velocity_record,label='Upper Angular Velocity',color='blue',alpha=0.5)
plt.plot(x,theta2_integral_record,label='Integrals',color='magenta')

plt.xlabel('Steps', fontsize=30)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#plt.ylim(-0.1,0.1)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.savefig('double_RL_upper.png')


#plt.plot(x,theta2_record,label='Upper Angle',color='black')
#plt.plot(x,theta2_velocity_record,label='Upper Angular Velocity',color='blue',alpha=0.5)
#plt.plot(x,theta2_integral_record,label='Integrals',color='magenta')

#plt.xlabel('Steps', fontsize=30)
#plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 16})
#plt.ylim(-0.1,0.1)
#plt.savefig('double_RL.png')
#plt.show()

plt.figure(figsize=(13,8.5))
plt.plot(x,actions_record,label='RL Actions',color='royalblue',alpha=0.5)
plt.xlabel('Steps', fontsize=30)
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 30})
#plt.ylim(-0.1,0.1)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.savefig('double_RL_actions.png')
#plt.show()
RL.close()
environment.close()
#env.close()



