from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number
from Normal import exploration
from Normal import measure_length
from Normal import moving_average
from Normal import prohibition_parameter
from Normal import prohibition_position
from Normal import environment

#training and evaluation with boundary
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))
evaluation_reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),evaluation_episode_number))

thigh_actuator_kp=[-3,-2]
thigh_actuator_kd=[-2,1]
leg_actuator_kp=[-.5,-0.5]
leg_actuator_kd=[-.5,0.2]
foot_actuator_kp=[-.5, 1]
foot_actuator_kd=[-.5,-0.1]
left_thigh_actuator_kp=[-3,-2]
left_thigh_actuator_kd=[-2,1]
left_leg_actuator_kp=[-.5,-0.5]
left_leg_actuator_kd=[-.5,0.2]
left_foot_actuator_kp=[-0.5, 1]
left_foot_actuator_kd=[-0.5,-0.1]

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
                y_position=states[1]

                rooty=states[1]
                velocity_rooty=states[10]

                thigh_angle=states[2]
                thigh_angular_velocity=states[11]

                leg_angle=states[3]
                leg_angular_velocity=states[12]

                foot_angle=states[4]
                foot_angular_velocity=states[13]

                left_thigh_angle=states[5]
                left_thigh_angular_velocity=states[14]

                left_leg_angle=states[6]
                left_leg_angular_velocity=states[15]

                left_foot_angle=states[7]
                left_foot_angular_velocity=states[16]
                actions = agent.act(states=states)
                if abs(y_position)>=prohibition_position[k]:
                    thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty
                    leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty
                    foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty
                    left_thigh_actions = left_thigh_actuator_kp[0]*rooty+left_thigh_actuator_kd[0]*velocity_rooty
                    left_leg_actions = left_leg_actuator_kp[0]*rooty+left_leg_actuator_kd[0]*velocity_rooty
                    left_foot_actions = left_foot_actuator_kp[0]*rooty+left_foot_actuator_kd[0]*velocity_rooty
                    actions=[thigh_actions,leg_actions,foot_actions,left_thigh_actions,left_leg_actions,left_foot_actions]                                   
                    states, terminal, reward = environment.execute(actions=actions)
                    reward= -1
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
environment.close()