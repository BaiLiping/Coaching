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
from Normal import measure_length
from Normal import moving_average
from Normal import environment

reward_record=np.zeros(episode_number)
evaluation_reward_record=np.zeros(evaluation_episode_number)

kp=[-0.6, -3]
kd=[-0.47, -0.7]

reward_record=[]
theta1_batch=[0]
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
        theta1_old=theta1_batch[count]
        count+=1
        sintheta1=states[1]
        sintheta2=states[2]
        velocity_theta1=states[6]
        velocity_theta2=states[7]
        theta1_batch.append(sintheta1)

        if abs(sintheta1)<=0.07:
            if abs(sintheta1)-abs(theta1_old)>0 and sintheta1*theta1_old>0:
                old_count=consecutive[cons_position]
                consecutive.append(count)
                cons_position+=1
                if old_count==count-1:
                    positive+=1
                else:
                    positive=1

            if positive==2:
                positive=0
                intervention=kp[0]*states[1]+kp[1]*states[2]+kd[0]*states[6]+kd[1]*states[7]
                print('intervention:',intervention)
                states, terminal, reward = environment.execute(actions=intervention)
                if terminal == 1:
                    break
        
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        episode_reward+=reward
    reward_record.append(episode_reward)


#evaluate
episode_reward = 0.0
eva_reward_record=[]
for j in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    eva_reward_record.append(episode_reward)
evaluation_reward_record=eva_reward_record
agent.close()
#save data
pickle.dump(reward_record, open( "double_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "double_evaluation_record.p", "wb"))
environment.close()
