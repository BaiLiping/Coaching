from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
episode_number=40000
average_over=100
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
'''
# Intialize reward record and set parameters

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0,-1,-3,-5]
prohibition_position=[0.2,0.3,0.4]


reward_record_without=pickle.load(open( "double_without_record.p", "rb"))
reward_record=pickle.load(open( "double_angle_record.p", "rb"))


#plot results
color_scheme=['green','orange','red','blue','yellowgreen','magenta','cyan']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    plt.figure(figsize=(20,10))
    plt.plot(x,reward_record_without,label='without prohibitive boundary',color='black')
    for j in range(len(prohibition_parameter)):
        plt.plot(x,reward_record[i][j],label='position '+str(prohibition_position[i])+' parameter '+str(prohibition_parameter[j]),color=color_scheme[j])
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
    plt.savefig('double_with_boundary_at_%s_plot.png' %prohibition_position[i])
