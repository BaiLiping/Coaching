from tensorforce import Agent, Environment
import tensorforce
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm


from normal import episode_number
from normal import average_over
from normal import evaluation_episode_number
from normal import exploration
from normal import measure_length
from normal import moving_average
from normal import prohibition_parameter
from normal import prohibition_position
from normal import theta_threshold_radians

from normal import environment
from normal import input_size
from normal import output_size
from normal import layers

if __name__ == "__main__":
    reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
    reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
    evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))
    for k in range(len(prohibition_position)):
        #training
        for i in range(len(prohibition_parameter)):
            record=[]
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

    environment.close()
