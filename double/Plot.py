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


#load data
double_without=pickle.load(open( "double_without_record.p", "rb"))
double_record=pickle.load(open( "double_record.p", "rb"))
double_evaluation_record_without=pickle.load(open( "double_evaluation_without_record.p", "rb"))
double_evaluation_record=pickle.load(open( "double_evaluation_record.p", "rb"))
evalu_without_ave=sum(double_evaluation_record_without)/evaluation_episode_number
evalu_ave=sum(double_evaluation_record)/evaluation_episode_number


#plot training results

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
double_without_average=moving_average(double_without,average_over)
double_record_average=moving_average(double_record,average_over)

fig=plt.figure(figsize=(13,7))
env_standard=800
x=range(len(double_record_average))
plt.plot(x,double_without_average,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
plt.plot(x,double_record_average,label='Coached by PID Controller\nEvaluation %s'%evalu_ave,color='magenta')
plt.xlabel('Episode Number', fontsize='large')
plt.ylabel('Episode Reward', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 18})
plt.axhline(y=env_standard, color='black', linestyle='dotted')
plt.savefig('double.png')