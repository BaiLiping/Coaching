from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=1000
average_over=20
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='LunarLander-v2')
'''
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Do Nothing
        1     Fire Left Engine
        2     Fire Main Engine
        3     Fire Right Engine

    Observation: Box(-np.inf, np.inf, shape=(8,))

        Type: Box(8)
        Num     Observation
        0       (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
        1       (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)
        2       vel.x*(VIEWPORT_W/SCALE/2)/FPS
        3       vel.y*(VIEWPORT_H/SCALE/2)/FPS
        4       Lander Angle
        5       20.0*self.lander.angularVelocity/FPS
        6       Legs[0] Contact with ground
        7       Legs[1] Contact with gound

    Terminal State:
        abs(state[0]) >= 1.0

    Prohibitive Boundary:
       the boundary set around abs(state[0])=0.95
           when x position is greater than 0.05, action 3
           when x position is less than -0.05, action 1
       the angle at abs(22) 0.4 radius
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[-15,-20,-25,-30]
prohibition_position=[0.5,0.9]

reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
theta_threshold_radians=0.4
#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='agent.json', environment=environment)
print('running experiment without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal=False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    record.append(episode_reward)
temp=np.array(record)
reward_record_without=moving_average(temp,average_over)
pickle.dump(reward_record_without, open( "lander_without_record.p", "wb"))



#with boundary
for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                angle=states[4]
                if angle>=prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=3
                elif angle<=-prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                else:
                    actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                episode_reward+=reward
            record.append(episode_reward)
        temp=np.array(record)
        reward_record[k][i]=moving_average(temp,average_over)

#save data
pickle.dump( reward_record, open( "lander_record.p", "wb"))

#plot results
color_scheme=['green','orange','red','blue','yellowgreen','magenta','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('lander_with_boundary_at_%s_plot.png' %prohibition_position[i])


agent.close()
environment.close()
