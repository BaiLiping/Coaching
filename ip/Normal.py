
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
environment = Environment.create(environment='gym', level='InvertedPendulum-v2')
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

if __name__ == "__main__":

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
    pickle.dump(reward_record_without, open( "ip_without_record.p", "wb"))

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
    pickle.dump(evaluation_reward_record_without, open( "ip_evaluation_without_record.p", "wb"))
    agent_without.close()
    environment.close()