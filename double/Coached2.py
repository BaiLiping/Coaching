from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number
from Normal import exploration
from Normal import environment

reward_record=[]
evaluation_reward_record=[]

kp=[-0.5,-2.9]
kd=[-0.5,-0.6]

reward_record=[]
agent = Agent.create(agent='agent.json', environment=environment)
for epi in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal = False
    count=0
    positive=0
    consecutive=[0]
    cons_position=0

    while not terminal:
        theta1_old=states[6]
        count+=1

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        episode_reward+=reward
        if terminal == 1:
            break

        sintheta1=states[1]
        sintheta2=states[2]
        velocity_theta1=states[6]
        velocity_theta2=states[7]


        if abs(velocity_theta1)>=0.1 and abs(velocity_theta1)<=0.4:
            if abs(velocity_theta1)-abs(theta1_old)>0 and velocity_theta1*theta1_old>0:
                old_count=consecutive[cons_position]
                consecutive.append(count)
                cons_position+=1
                if old_count==count-1:
                    positive+=1
                else:
                    positive=1

            if positive==2:
                print('before:',states[6])
                intervention=kp[0]*states[1]+kp[1]*states[2]+kd[0]*states[6]+kd[1]*states[7]
                print('intervention:',intervention)
                states, terminal, reward = environment.execute(actions=intervention)
                print('after:', states[6])
    reward_record.append(episode_reward)

#evaluate
episode_reward = 0.0
for j in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record.append(episode_reward)
agent.close()
#save data
pickle.dump(reward_record, open( "double_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "double_evaluation_record.p", "wb"))
environment.close()
