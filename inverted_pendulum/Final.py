from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#setparameters
num_steps=50 #update exploration rate over n steps
initial_value=0.95 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=200
evaluation_episode_number=5
average_over=20

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='InvertedPendulum-v2')
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

prohibition_parameter=[0]
prohibition_position=[0.15,0.17]
'''
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
'''
reward_record_without_average=pickle.load(open( "without_average_record.p", "rb"))
reward_record_without=pickle.load(open( "without_record.p", "rb"))
evaluation_reward_record_without=pickle.load(open( "evaluation_without_record.p", "rb"))

kp=25
kd=3
#training with coaching
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))
for k in range(len(prohibition_position)):
    #training
    for i in range(len(prohibition_parameter)):
        record=[]
        angle_record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('training agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                theta=states[1]
                angular_velocity=states[3]
                theta_states=[theta,angular_velocity]
                actions = agent.act(states=states)
                if theta>=prohibition_position[k]:
                    #print('angular_velocity: ',angular_velocity)
                    #x=np.array(theta_states)
                    #x_ = PolynomialFeatures(degree=6, include_bias=True).fit_transform([x])
                    #actions_predict=model.predict(x_)
                    #actions_predict=np.clip(actions_predict.copy(), -5, 5)
                    #reward-=abs(actions_predict)
                    #print('actions_predict',actions_predict)
                    #states, terminal, reward = environment.execute(actions=actions_predict)
                    #actions=5
                    actions=kp*theta+kd*angular_velocity
                    states, terminal, reward = environment.execute(actions=actions)
                    #states[1]=theta*0.09
                    #states[3]= 0
                    reward=-1
                    episode_reward+=reward
                    agent.observe(terminal=terminal, reward=reward)
                elif theta<=-prohibition_position[k]:
                    #actions=-5
                    actions=kp*theta+kd*angular_velocity
                    states, terminal, reward = environment.execute(actions=actions)
                    #states[1]=theta*0.09
                    #states[3]= 0
                    reward=-1
                    episode_reward+=reward
                    agent.observe(terminal=terminal, reward=reward)
                else:
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)
                    episode_reward+=reward
                #print('action ',actions)
            record.append(episode_reward)
            #print(episode_reward)
        reward_record[k][i]=record
        temp=np.array(record)
        reward_record_average[k][i]=moving_average(temp,average_over)
        x_angle=range(len(angle_record))
        plt.figure(figsize=(30,10))
        plt.plot(x_angle,angle_record)
        plt.savefig('angle%s%s.png' %(k,i))

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