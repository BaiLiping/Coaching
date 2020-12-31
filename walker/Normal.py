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

episode_number=10000
evaluation_episode_number=50
average_over=100

# Pre-defined or custom environment
environment = Environment.create(environment='gym', level='Walker2d-v3')
'''
For detailed notes on how to interact with the Mujoco environment, please refer
to note https://bailiping.github.io/Mujoco/

Observation:
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    Num    Observation                                 Min            Max
           rootx(_get_obs states from  root z)          Not Limited
    0      rootz                                        Not Limited
    1      rooty                                        Not Limited
    2      thigh joint                                 -150           0
    3      leg joint                                   -150           0
    4      foot joint                                  -45            45
    5      thigh left joint                            -150           0
    6      leg left joint                              -150           0
    7      foot left joint                             -45            45
    8      velocity of rootx                           -10            10
    9      velocity of rootz                           -10            10
    10     velocity of rooty                           -10            10
    11     angular velocity of thigh joint             -10            10
    12     angular velocity of leg joint               -10            10
    13     angular velocity of foot joint              -10            10
    14     angular velocity of thigh left joint        -10            10
    15     angular velocity of leg left joint          -10            10
    16     angular velocity of foot left joint         -10            10

Actions:
    0     Thigh Joint Motor                             -1             1
    1     Leg Joint Motor                               -1             1
    2     Foot Joint Motor                              -1             1
    3     Thigh Left Joint Motor                        -1             1
    4     Leg Left Joint Motor                          -1             1
    5     Foot Left Joint Motor                         -1             1
Termination:
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
'''

# Intialize reward record and set parameters
#define the length of the vector

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[-1]
prohibition_position=[0.7]



#compare to agent trained without prohibitive boundary
#training of agent without prohibitive boundary
if __name__ == "__main__":
    reward_record_without=[]

    agent_without = Agent.create(agent='agent.json', environment=environment,exploration=exploration)
    states=environment.reset()
    terminal = False
    print('training agent without boundary')
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
        print(episode_reward)
    temp=np.array(reward_record_without)
    reward_record_without_average=moving_average(temp,average_over)
    pickle.dump(reward_record_without_average, open( "without_average_record.p", "wb"))
    pickle.dump(reward_record_without, open( "without_record.p", "wb"))

    #plot
    x=range(len(measure_length))
    plt.figure(figsize=(20,10))
    plt.plot(x,reward_record_without_average,label='without prohibitive boundary',color='black')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('plot.png')


    #evaluate the agent without Boundary
    episode_reward = 0.0
    evaluation_reward_record_without=[]
    print('evaluating agent without boundary')
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
        print(evaluation_reward_record_without)
    pickle.dump(evaluation_reward_record_without, open( "evaluation_without_record.p", "wb"))
    agent_without.save(directory='model1', format='numpy')
    agent_without.close()