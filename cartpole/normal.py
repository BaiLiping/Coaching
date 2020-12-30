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

environment = Environment.create(environment='gym', level='CartPole-v1', max_episode_timesteps=500)
input_size=environment.states()['shape'][0]
output_size=1
layers=[
        dict(type='dense',size=input_size,    activation='relu'),
        dict(type='dense',size=input_size*2,  activation='relu'),
        dict(type='dense',size= output_size*3,activation='relu'),
        dict(type='dense',size=output_size)
       ]

if __name__ == "__main__":
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
    agent.close()
    environment.close()
