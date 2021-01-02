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


#load data
ip_without=pickle.load(open( "ip_without_record.p", "rb"))
ip_record=pickle.load(open( "ip_record.p", "rb"))
ip_evaluation_record_without=pickle.load(open( "ip_evaluation_without_record.p", "rb"))
ip_evaluation_record=pickle.load(open( "ip_evaluation_record.p", "rb"))
evalu_without_ave=sum(ip_evaluation_record_without)/evaluation_episode_number
evalu_ave=sum(ip_evaluation_record)/evaluation_episode_number

#smooth_over
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
ip_without_average=moving_average(ip_without,average_over)
ip_record_average=moving_average(ip_record,average_over)

#plot
fig=plt.figure(figsize=(13,7))
env_standard=800
x=range(len(ip_record_average))
plt.plot(x,ip_without_average,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
plt.plot(x,ip_record_average,label='Coached by PID Controller\nEvaluation %s'%evalu_ave,color='magenta')
plt.xlabel('Episode Number', fontsize='large')
plt.ylabel('Episode Reward', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 18})
plt.axhline(y=env_standard, color='black', linestyle='dotted')
plt.savefig('ip.png')