from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

from AutoTune import episode_number
from AutoTune import average_over
from AutoTune import evaluation_episode_number

from AutoTune import exploration


if __name__ == "__main__":

    #training single action agent
    environment = Environment.create(environment='gym', level='InvertedPendulumPID-v0')
    reward_record_single=[]

    agent_p = Agent.create(agent='agentpid.json', environment=environment,exploration=exploration)
    agent_i= Agent.create(agent='agentpid.json',environment=environment,exploration=exploration)
    agent_d = Agent.create(agent='agentpid.json',environment=environment,exploration=exploration)

    print('Training MultiAgents')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment.reset()
        terminal= False
        integral=0
        while not terminal:
            states[4]=0.0
            actions_p = agent_p.act(states=states)
            states[4]=actions_p[0]
            states[5]=0.0
            actions_i = agent_i.act(states=states)
            states[5]=actions_i[0]
            states[6]=0.0
            actions_d = agent_d.act(states=states)
            states[6] = actions_d[0]

            slider=states[0]
            hinge=states[1]
            slider_velocity=states[2]
            hinge_velocity=states[3]
            integral+=hinge

            actions=hinge*actions_p[0]+hinge_velocity*actions_d[0]+integral*actions_i[0]
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward+=reward
            agent_p.observe(terminal=terminal, reward=reward)
            agent_i.observe(terminal=terminal, reward=reward)
            agent_d.observe(terminal=terminal, reward=reward)
        reward_record_single.append(episode_reward)
        print(episode_reward)
    temp=np.array(reward_record_single)
    reward_record_single_average=moving_average(temp,average_over)
    pickle.dump(reward_record_single_average, open( "pid_single_average_record.p", "wb"))
    pickle.dump(reward_record_single, open( "pid_single_record.p", "wb"))