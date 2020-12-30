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

model=pickle.load(open('regression_model.sav', 'rb'))
#kp=[-0.47832891*5,-2.93581233*5]
#kd=[-0.47017734,-0.70055663]
#kp=[-0.47832891,-2.93581233]
#kd=[-0.47017734,-0.70055663]
kp=[-0.54124971, -3.05534616]
kd=[-0.47012709, -0.70023993]

# polynomial controller
environment_control = gym.make('InvertedDoublePendulum-v2')
episode_reward=0
states = environment_control.reset()
terminal=False
theta_states=[]
while not terminal:
    #theta_states=[states[1],states[2],states[6],states[7]]
    #x=np.array(theta_states)
    #x_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform([x])
    #actions_predict= model.predict(x_)
    #states, reward, terminal,info = environment_control.step(actions_predict)
    #episode_reward+=reward
    actions_predict=kp[0]*states[0]+kp[1]*states[2]+kd[0]*states[6]+kd[1]*states[7]
    print('actions',actions_predict)
    states, reward, terminal,info = environment_control.step(actions_predict)
    episode_reward+=reward


print(episode_reward)