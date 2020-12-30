from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

episode_number=200
evaluation_episode_number=5
average_over=20
length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[0]
prohibition_position=[0.19,0.2]

reward_record_without=pickle.load(open( "without_record.p", "rb"))
reward_record_without_average=pickle.load(open( "without_average_record.p", "rb"))
evaluation_reward_record_without=pickle.load(open( "evaluation_without_record.p", "rb"))
reward_record=pickle.load(open("record.p", "rb"))
reward_record_average=pickle.load(open("average_record.p","rb"))
evaluation_reward_record=pickle.load(open( "evaluation_record.p", "rb"))
average_without=sum(evaluation_reward_record_without)/evaluation_episode_number

#plot training results
color_scheme=['magenta','yellowgreen','orange','blue','red','cyan','green']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    fig=plt.figure(figsize=(13,7))
    #plt.text(20, 230, 'Protective Boundadry \n at Position %s' %(prohibition_position[i]),fontsize=14)
    plt.plot(x,reward_record_without_average,label='Normal Training',color='black',linewidth=3,linestyle='-.')
    for j in range(len(prohibition_parameter)):
        average=0
        average=(sum(evaluation_reward_record[i][j])/evaluation_episode_number)
        plt.plot(x,reward_record_average[i][j],label='With PID Controller as Coach',color=color_scheme[j])
    plt.xlabel('Episode Number', fontsize='large')
    plt.ylabel('Episode Reward', fontsize='large')
    plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 10})
    plt.axhline(y=800, color='black', linestyle='dotted')
    plt.savefig('Inverted_Pendulum_with_Boundary_at_%s.png' %prohibition_position[i])
