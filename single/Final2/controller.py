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

model=pickle.load(open('coaching_model.sav', 'rb'))

# polynomial controller
environment_control = gym.make('InvertedPendulumBLP-v2')
episode_reward=0
states = environment_control.reset()
terminal=False
theta_states=[]
while not terminal:
    theta_states=[states[1],states[3]]
    print('states:',theta_states)
    x=np.array(theta_states)
    x_ = PolynomialFeatures(degree=6, include_bias=True).fit_transform([x])
    actions_predict= model.predict(x_)
    print('action: ',actions_predict)
    actions_predict=np.clip(actions_predict.copy(), -10, 10)
    states, reward, terminal,info = environment_control.step(actions_predict)
    episode_reward+=reward

print(episode_reward)