from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=400
average_over=20
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole-v1', max_episode_timesteps=1000)
'''
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Terminal State:
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-5,-10,-15,-20,-25,-30]
prohibition_position=[0.1,0.3,0.5,0.7,0.9,0.95,0.99]


reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
theta_threshold_radians=12*2*math.pi/360
x_threshold=2.4

for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)-1):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                position=states[0]
                angle=states[2]
                if angle>=prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=0
                elif angle<=-prohibition_position[k]*theta_threshold_radians:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                else:
                    if position>=prohibition_position[k]*x_threshold:
                        episode_reward+=prohibition_parameter[i]
                        actions=agent.act(states=states)
                        actions=0
                    elif position<=-prohibition_position[k]*x_threshold:
                        episode_reward+=prohibition_parameter[i]
                        actions=agent.act(states=states)
                        actions=1
                    else:
                        actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                episode_reward+=reward
            record.append(episode_reward)
        temp=np.array(record)
        reward_record[k][i]=moving_average(temp,average_over)
#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='agent.json', environment=environment) 
states=environment.reset()
terminal = False
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
for k in range(len(prohibition_position)):
    reward_record[k][len(prohibition_parameter)-1]=moving_average(temp,average_over)
#plot results
color_scheme=['green','orange','red','yellow','yellowgreen','black']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(loc="upper left")
    plt.savefig('cartpole_with_both_boundary_at_%s_plot.png' %prohibition_position[i])
agent.close()
environment.close() 
