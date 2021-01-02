from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

#setparameters
num_steps=100 #update exploration rate over n steps
initial_value=0.9 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=2500
evaluation_episode_number=5
average_over=100

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='InvertedDoublePendulum-v2')
'''
    Observation:

        Num    Observation               Min            Max
        0      x_position
        1      sin(theta1)
        2      sin(theta2)
        3      cos(theta1)
        4      cos(theta2)
        5      velocity of x
        6      velocity of theta1
        7      velocity of theta2
        8      constraint force on x
        9      constraint force on theta1
        10     constraint force on theta2

    Action: (-1,1) actuation on the cart continuous
    Terminal State:
        y<=1 can not be observed directly
    Boundary:
       The most important element for keeping the balance of a double
       pendulum is the angle between the two poles. Observation has
       data on sin(theta1) and sin(theta2) and from there we can
       derive the actual angle between the poles (math.pi-theta1+theta2)
       then math.pi-(math.pi-theta1+theta2)=theta1-theta2 which is the
       information on how pointing the angle is. We want theta1-theta2
       to be as small as possible
    reward function:
        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()
            x, _, y = self.sim.data.site_xpos[0]
            dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
            v1, v2 = self.sim.data.qvel[1:3]
            vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
            alive_bonus = 10
            r = alive_bonus - dist_penalty - vel_penalty
            done = bool(y <= 1)
            return ob, r, done, {}
        def _get_obs(self):
            return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
            ]).ravel()
'''

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)


if __name__ == "__main__":
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
    pickle.dump(reward_record_without, open( "double_without_record.p", "wb"))

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
            actions, internals = agent_without.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_without.append(episode_reward)
    pickle.dump(evaluation_reward_record_without, open( "double_evaluation_without_record.p", "wb"))
