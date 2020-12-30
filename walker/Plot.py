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

reward_record_without=pickle.load(open( "without_record.p", "rb"))
reward_record_without_average=pickle.load(open( "without_average_record.p", "rb"))
evaluation_reward_record_without=pickle.load(open( "evaluation_without_record.p", "rb"))
reward_record=pickle.load(open("record.p", "rb"))
reward_record_average=pickle.load(open("average_record.p","rb"))
evaluation_reward_record=pickle.load(open( "evaluation_record.p", "rb"))
average_without=sum(evaluation_reward_record_without)/evaluation_episode_number

#plot training results
color_scheme=['yellowgreen','magenta','orange','blue','red','cyan','green']
x=range(len(measure_length))
for i in range(len(prohibition_position)):
    fig=plt.figure(figsize=(15,7))
    plt.text(1000, 600, 'Protective Boundadry \n at Position %s' %(prohibition_position[i]),fontsize=14)
    plt.plot(x,reward_record_without_average,label='Normal Training \n Evaluation Average: \n %s' %average_without,color='black',linewidth=2,linestyle='-.')
    for j in range(len(prohibition_parameter)):
        average=0
        average=(sum(evaluation_reward_record[i][j])/evaluation_episode_number)
        plt.plot(x,reward_record_average[i][j],label='Parameter %s \n Evaluation Average: \n %s' %(prohibition_parameter[j],average),color=color_scheme[j])
    plt.xlabel('Episode Number', fontsize='large')
    plt.ylabel('Episode Reward', fontsize='large')
    plt.axhline(y=800, color='black', linestyle='dotted')
    plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 11})
    plt.savefig('Walker_with_Boundary_at_%s.png' %prohibition_position[i])
