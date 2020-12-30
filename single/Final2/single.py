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

episode_number=100
evaluation_episode_number=5
average_over=10

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='InvertedPendulum-v2', max_episode_timesteps=300)
'''
    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}
   def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
    <motor gear="100" joint="slider" name="slide"/> unboundede!
'''
# Intialize reward record and set parameters
#define the length of the vector

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-5,-7,-10,-12,-15]
prohibition_position=[0.1,0.3,0.6]

#compare to agent trained without prohibitive boundary

#training of agent without prohibitive boundary
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
temp=np.array(reward_record_without)
reward_record_without_average=moving_average(temp,average_over)
pickle.dump(reward_record_without_average, open( "without_average_record.p", "wb"))
pickle.dump(reward_record_without, open( "without_record.p", "wb"))

#plot
x=range(len(measure_length))
plt.figure(figsize=(10,10))
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
        actions, internals = agent_without.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_without.append(episode_reward)
pickle.dump(evaluation_reward_record_without, open( "evaluation_without_record.p", "wb"))
agent_without.close()

#training and evaluation with boundary
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))

for k in range(len(prohibition_position)):
    #training
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('training agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                x_position=states[0]
                theta=states[1]
                actions = agent.act(states=states)
                if theta>=prohibition_position[k]:
                    actions=1
                    states, terminal, reward = environment.execute(actions=actions)
                    reward+= prohibition_parameter[i]
                    states=[x_position,0.9*prohibition_position[k],0,0]
                    episode_reward+=reward
                    agent.observe(terminal=terminal, reward=reward)
                elif theta<=-prohibition_position[k]:
                    actions=-1
                    states, terminal, reward = environment.execute(actions=actions)
                    reward+= prohibition_parameter[i]
                    states=[x_position,-0.9*prohibition_position[k],0,0]
                    episode_reward+=reward
                    agent.observe(terminal=terminal, reward=reward)
                else:
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)
                    episode_reward+=reward
            record.append(episode_reward)
        reward_record[k][i]=record
        temp=np.array(record)
        reward_record_average[k][i]=moving_average(temp,average_over)

        #evaluate
        episode_reward = 0.0
        eva_reward_record=[]
        print('evaluating agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
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
        evaluation_reward_record[k][i]=eva_reward_record
        agent.close()
#save data
pickle.dump(reward_record, open( "record.p", "wb"))
pickle.dump(reward_record_average, open( "average_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "evaluation_record.p", "wb"))


#plot training results
color_scheme=['yellowgreen','magenta','orange','blue','red','cyan','green']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure(figsize=(10,10))
    plt.plot(x,reward_record_without_average,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record_average[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('Inverted_pendulum_with_boundary_at_%s_plot.png' %prohibition_position[i])

#indicate evaluation results
average_without=sum(evaluation_reward_record_without)/evaluation_episode_number
print("the average of agent trained without boundary is %s" %average_without)
average=0
for i in range(len(prohibition_position)):
    for j in range(len(prohibition_parameter)):
        average=(sum(evaluation_reward_record[i][j])/evaluation_episode_number)
        print("the average of agent trained with boundary at %s with parameter %s is %s" %(prohibition_position[i],prohibition_parameter[j],average))

environment.close()
