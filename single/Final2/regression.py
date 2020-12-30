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


environment = Environment.create(environment='gym', level='InvertedPendulumBLP-v2')

#polynomial regression
coach= Agent.load(directory='model5', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_states=[]
for k in range(30):
    states = environment.reset()
    terminal=False
    while not terminal:
    	theta_states.append([states[1],states[3]])
    	actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
    	states, terminal, reward = environment.execute(actions=actions)
    	actions_record.append(actions)

x, y = np.array(theta_states), np.array(actions_record)
x_ = PolynomialFeatures(degree=6, include_bias=True).fit_transform(x)
model = LinearRegression().fit(x_, y)
pickle.dump(model, open('coaching_model.sav', 'wb'))