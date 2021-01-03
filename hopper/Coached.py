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
from Normal import environment

reward_record=[]
evaluation_reward_record=[]

#set PID parameters
thigh_actuator_kp=[-2,-2,-0.5,-1]
thigh_actuator_kd=[-2,1, 0.2,-0.4]
leg_actuator_kp=[-0.4,-0.5,-0.1,-0.2]
leg_actuator_kd=[-1,0.2,-1,-0.1]
foot_actuator_kp=[-2, 1, 0.5, -1]
foot_actuator_kd=[-0.4,-0.1,-0.1,-0.5]

reward_record=[]

agent = Agent.create(agent='agent.json', environment=environment)
print('Training Agent with PID Coaching')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal = False
    count=0
    positive=0
    consecutive=[0]
    velocity_batch=[0]
    cons_position=0

    while not terminal:
        velocity_old=velocity_batch[count]
        count+=1
        velocity_batch.append(states[7])
        rooty=states[1]
        velocity_rooty=states[7]
        thigh_angle=states[2]
        thigh_angular_velocity=states[8]
        leg_angle=states[3]
        leg_angular_velocity=states[9]
        foot_angle=states[4]
        foot_angular_velocity=states[10]

        if abs(states[7])<=0.4 and abs(states[7])>=0.1:
            if abs(states[7])-abs(velocity_old)>0 and states[7]*velocity_old>0:
                old_count=consecutive[cons_position]
                consecutive.append(count)
                cons_position+=1
                if old_count==count-1:
                    positive+=1
                else:
                    positive=1

            if positive==3:
                positive=0
                thigh_actions = thigh_actuator_kp[0]*rooty+thigh_actuator_kd[0]*velocity_rooty+thigh_actuator_kp[1]*thigh_angle+thigh_actuator_kd[1]*thigh_angular_velocity+thigh_actuator_kp[2]*leg_angle+thigh_actuator_kd[2]*leg_angular_velocity+thigh_actuator_kp[3]*foot_angle+thigh_actuator_kd[3]*foot_angular_velocity
                leg_actions = leg_actuator_kp[0]*rooty+leg_actuator_kd[0]*velocity_rooty+leg_actuator_kp[1]*thigh_angle+leg_actuator_kd[1]*thigh_angular_velocity+leg_actuator_kp[2]*leg_angle+leg_actuator_kd[2]*leg_angular_velocity+leg_actuator_kp[3]*foot_angle+leg_actuator_kd[3]*foot_angular_velocity
                foot_actions = foot_actuator_kp[0]*rooty+foot_actuator_kd[0]*velocity_rooty+foot_actuator_kp[1]*thigh_angle+foot_actuator_kd[1]*thigh_angular_velocity+foot_actuator_kp[2]*leg_angle+foot_actuator_kd[2]*leg_angular_velocity+foot_actuator_kp[3]*foot_angle+foot_actuator_kd[3]*foot_angular_velocity
                intervention=[thigh_actions,leg_actions,foot_actions]
                print('intervention:',intervention)
                states, terminal, reward = environment.execute(actions=intervention)
                if terminal == 1:
                    continue
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        episode_reward+=reward
    reward_record.append(episode_reward)

#evaluate
print('Evaluating Agent with PID Coaching')
episode_reward = 0.0
for j in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record.append(episode_reward)
agent.close()
#save data
pickle.dump(reward_record, open( "hopper_record.p", "wb"))
pickle.dump(evaluation_reward_record, open( "hopper_evaluation_record.p", "wb"))
environment.close()