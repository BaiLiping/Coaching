from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm


#setparameters
num_steps=1000 #update exploration rate over n steps
initial_value=0.9 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=3000
evaluation_episode_number=50
average_over=100

# Pre-defined or custom environment
environment = Environment.create(environment='gym', level='Hopper-v3')
'''
For detailed notes on how to interact with the Mujoco environment, please refer
to note https://bailiping.github.io/Mujoco/

Observation:

    Num    Observation                                 Min            Max
           x_position(exclude shown up in info instead) Not Limited
    0      rootz                                        Not Limited
    1      rooty                                        Not Limited
    2      thigh joint                                  -150           0
    3      leg joint                                    -150           0
    4      foot joint                                   -45           45
    5      velocity of rootx                           -10            10
    6      velocity of rootz                           -10            10
    7      velocity of rooty                           -10            10
    8      angular velocity of thigh joint             -10            10
    9      angular velocity of leg joint               -10            10
    10     angular velocity of foot joint              -10            10

Actions:
    0     Thigh Joint Motor                             -1             1
    1     Leg Joint Motor                               -1             1
    2     Foot Joint Motor                              -1             1
Termination:
    healthy_angle_range=(-0.2, 0.2)
'''

if __name__ == "__main__":

    reward_record_without=[]
    agent_without = Agent.create(agent='agent.json', environment=environment,exploration=exploration)
    states=environment.reset()
    terminal = False
    print('Training Normal Agent')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment.reset()
        terminal= False
        while not terminal:
            actions = agent_without.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward+=reward
            agent_without.observe(terminal=terminal, reward=reward)
        reward_record_without.append(episode_reward)
    pickle.dump(reward_record_without, open( "hopper_without_record.p", "wb"))

    #evaluate the agent without Boundary
    episode_reward = 0.0
    evaluation_reward_record_without=[]
    print('Evaluating Normal Agent')
    for _ in tqdm(range(evaluation_episode_number)):
        episode_reward=0
        states = environment.reset()
        internals = agent_without.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent_without.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_without.append(episode_reward)
    pickle.dump(evaluation_reward_record_without, open( "hopper_evaluation_without_record.p", "wb"))
    agent_without.close()