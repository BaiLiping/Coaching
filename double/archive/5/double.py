from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

#define episode and average_over
episode_number=60000
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

    Action: (-1,1) actuation on the cart
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
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-1,-3,-5]
prohibition_position=[0.2,0.3,0.4]

#compare to agent trained without prohibitive boundary
record=[]
agent = Agent.create(agent='agent.json', environment=environment)
states=environment.reset()
terminal = False
print('running experiment without boundary')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment.reset()
    terminal= False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        episode_reward+=reward
        agent.observe(terminal=terminal, reward=reward)
    record.append(episode_reward)
temp=np.array(record)
reward_record_without=record
reward_record_without_average=moving_average(temp,average_over)
pickle.dump(reward_record_without_average, open( "double_without_average_record.p", "wb"))
pickle.dump(reward_record_without, open( "double_without_record.p", "wb"))

#with boundary
reward_record_average=np.zeros((len(prohibition_position),len(prohibition_parameter),len(measure_length)))
reward_record=np.zeros((len(prohibition_position),len(prohibition_parameter),episode_number))

for k in range(len(prohibition_position)):
    for i in range(len(prohibition_parameter)):
        record=[]
        agent = Agent.create(agent='agent.json', environment=environment)
        print('running experiment with boundary position at %s and prohibitive parameter %s' %(prohibition_position[k],prohibition_parameter[i]))
        for _ in tqdm(range(episode_number)):
            episode_reward=0
            states = environment.reset()
            terminal = False
            while not terminal:
                sintheta1=states[1]
                sintheta2=states[2]
                theta1=math.asin(sintheta1)
                theta2=math.asin(sintheta2)
                angle=theta1-theta2
                if angle>=prohibition_position[k]*math.pi:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=1
                elif angle<=-prohibition_position[k]*math.pi:
                    episode_reward+= prohibition_parameter[i]
                    actions = agent.act(states=states)
                    actions=-1
                else:
                    actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                episode_reward+=reward
            record.append(episode_reward)
        reward_record[k][i]=record
        temp=np.array(record)
        reward_record_average[k][i]=moving_average(temp,average_over)


#save data
pickle.dump( reward_record, open( "double_record.p", "wb"))
pickle.dump( reward_record_average, open( "double_average_record.p", "wb"))


#plot results
color_scheme=['green','orange','red','blue','yellowgreen','magenta','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')


#plot results
color_scheme=['yellowgreen','magenta','green','orange','red','blue','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure()
    plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('double_with_boundary_at_%s_plot.png' %prohibition_position[i])


agent.close()
environment.close()
