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

kp=15
kd=2.25
#training with coaching
reward_record=[]
evaluation_reward_record=[]

reward_record=[]
agent = Agent.create(agent='agent.json', environment=environment)
print('Training Agent with PID Coaching')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal = False
    count=0
    positive=0
    consecutive=[0]
    cons_position=0

    while not terminal:
        velocity_old=states[3]
        count+=1

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        episode_reward+=reward

        theta=states[1]
        angular_velocity=states[3]

        if terminal==1:
            break

        if abs(angular_velocity)<=0.3:
            if abs(angular_velocity)-abs(velocity_old)>0 and angular_velocity*velocity_old>0:
                old_count=consecutive[cons_position]
                consecutive.append(count)
                cons_position+=1
                if old_count==count-1:
                    positive+=1
                else:
                    positive=1

            if positive==2:
                positive=0
                print('before:' ,states[3])
                intervention=kp*theta+kd*angular_velocity
                print('coach intervention:',intervention)
                states, terminal, reward = environment.execute(actions=intervention)
                print('after:' ,states[3])
        
        
    reward_record.append(episode_reward)
    #print('reward:',episode_reward)

#evaluate
print('Evaluating Agent with PID Coaching')
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
pickle.dump(reward_record, open( "ip_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "ip_evaluation_record.p", "wb"))
environment.close()