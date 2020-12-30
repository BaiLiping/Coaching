from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm


#setparameters
num_steps=10 #update exploration rate over n steps
initial_value=0.95 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=200
evaluation_episode_number=5


# Pre-defined or custom environment
environment = Environment.create(environment='gym', level='InvertedPendulumBLP-v2')

length=np.zeros(episode_number)

reward_record_without=[]

agent_without = Agent.create(agent='agent.json', environment=environment,exploration=exploration)
states=environment.reset()
terminal = False
print('training agent without boundary')
angle_record=[]
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent_without.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        reward=0
        if abs(states[1])>=0.1:
            reward+=abs(states[1])*10
        #else:
        #    reward-=abs(states[1])*10
        #    reward-=abs(states[3])*10
        angle_record.append(states[1])
        agent_without.observe(terminal=terminal, reward=reward)
        episode_reward+=reward
    reward_record_without.append(episode_reward)
agent_without.save(directory='model5', format='numpy')
x=range(episode_number)
x_angle=range(len(angle_record))
plt.figure(figsize=(10,10))
plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
plt.savefig('plot5.png')

plt.figure(figsize=(30,10))
plt.plot(x_angle,angle_record)
plt.savefig('angle5.png')