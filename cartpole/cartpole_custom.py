from tensorforce import Agent, Environment
import tensorforce
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

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



#setparameters

episode_number=300
average_over=40
evaluation_episode_number=5

prohibition_parameter=[0,-5,-7,-10,-12,-15]
prohibition_position=[0.5,0.7,0.9]

theta_threshold_radians=12*2*math.pi/360

def set_exploration(num_steps,initial_value,decay_rate,set_type='exponential'):
    exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)
    return exploration
exploration=set_exploration(5,0.0,0.5)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)


environment = Environment.create(
    environment='gym', level='CartPole-v1', max_episode_timesteps=500)

input_size=environment.states()['shape'][0]
output_size=1

layers=[
        dict(type='dense',size=input_size,    activation='relu'),
        dict(type='dense',size=input_size*2,  activation='relu'),
        dict(type='dense',size= output_size*3,activation='relu'),
        dict(type='dense',size=output_size)
       ]
customized_agent=tensorforce.agents.ProximalPolicyOptimization(
                            states=environment.states(),
                            actions=environment.actions(),
                            max_episode_timesteps=500,
                            batch_size=10,
                            network=dict(type='custom',layers=layers),
                            use_beta_distribution='true',
                            memory='minimum',
                            update_frequency=1,
                            learning_rate=1e-3,
                            multi_step=5,
                            subsampling_fraction=0.91,
                            likelihood_ratio_clipping=0.09,
                            discount=1.0,
                            predict_terminal_values='false',
                            baseline=dict(type='custom',layers=layers),
                            baseline_optimizer=dict(optimizer="adam",learning_rate=1e-3,multi_step=5),
                            l2_regularization=0.0,
                            entropy_regularization=0.3,           
                            state_preprocessing='linear_normalization',
                            exploration=exploration,
                            variable_noise=0.0,
                            recorder=None,
                            parallel_interactions=1
                            )

agent=Agent.create(agent=customized_agent,environment=environment)
reward_record_without=[]
print('training agent without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    reward_record_without.append(episode_reward)
    print(episode_reward)
temp=np.array(reward_record_without)
reward_record_without_average=moving_average(temp,average_over)
pickle.dump(reward_record_without_average, open( "without_average_record.p", "wb"))
pickle.dump(reward_record_without, open( "without_record.p", "wb"))

#evaluate the agent without Boundary
episode_reward = 0.0
evaluation_reward_record_without=[]
print('evaluating agent without boundary')
for _ in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_without.append(episode_reward)
pickle.dump(evaluation_reward_record_without, open( "evaluation_without_record.p", "wb"))
print(evaluation_reward_record_without)

#training and evaluation with boundary
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))

for k in range(len(prohibition_position)):
    #training
    for i in range(len(prohibition_parameter)):
        record=[]
        agent=Agent.create(agent=customized_agent,environment=environment)
        print('training agent with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                x_position=states[0]
                angle=states[2]
                actions= agent.act(states=states)
                if angle>=prohibition_position[k]*theta_threshold_radians:
                    actions=1
                    states,terminal,reward=environment.execute(actions=actions)
                    states=[x_position,0,0.9*prohibition_position[k]*theta_threshold_radians,0]
                    reward+=prohibition_parameter[i]
                    episode_reward+=reward
                    agent.observe(terminal=terminal,reward=reward)
                elif angle<=-prohibition_position[k]*theta_threshold_radians:
                    actions=0
                    states,terminal,reward=environment.execute(actions=actions)
                    states=[x_position,0,0.9*-prohibition_position[k]*theta_threshold_radians,0]
                    reward+=prohibition_parameter[i]
                    episode_reward+=reward
                    agent.observe(terminal=terminal,reward=reward)
                else:
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)
                    episode_reward+=reward
            record.append(episode_reward)
            print(episode_reward)
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
    plt.savefig('cartpole_with_boundary_at_%s_plot.png' %prohibition_position[i])

#indicate evaluation results
average_without=sum(evaluation_reward_record_without)/evaluation_episode_number
print("the average of agent trained without boundary is %s" %average_without)
average=0
for i in range(len(prohibition_position)):
    for j in range(len(prohibition_parameter)):
        average=(sum(evaluation_reward_record[i][j])/evaluation_episode_number)
        print("the average of agent trained with boundary at %s with parameter %s is %s" %(prohibition_position[i],prohibition_parameter[j],average))

environment.close()
