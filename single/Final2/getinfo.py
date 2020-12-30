from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


environment = Environment.create(environment='gym', level='InvertedPendulum-v2')
coach= Agent.load(directory='model5', format='numpy')
internals = coach.initial_internals()
actions_record=[]
theta_state=[]
for k in range(30):
    states = environment.reset()
    terminal=False
    while not terminal:
        actions, internals = coach.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)                
        actions_record.append(actions)
        theta_state.append([states[1],states[3]])
        print('states',states)
        print('actions',actions)
    

x, y = np.array(theta_state), np.array(actions_record)
model = LinearRegression().fit(x, y)
print(model.coef_)
