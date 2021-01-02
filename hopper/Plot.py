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
hopper_without=pickle.load(open( "hopper_without_record.p", "rb"))
hopper_record=pickle.load(open( "hopper_record.p", "rb"))
hopper_evaluation_record_without=pickle.load(open( "hopper_evaluation_without_record.p", "rb"))
hopper_evaluation_record=pickle.load(open( "hopper_evaluation_record.p", "rb"))
evalu_without_ave=sum(hopper_evaluation_record_without)/evaluation_episode_number
evalu_ave=sum(hopper_evaluation_record)/evaluation_episode_number


#plot training results
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
hopper_without_average=moving_average(hopper_without,average_over)
hopper_record_average=moving_average(hopper_record,average_over)

fig=plt.figure(figsize=(13,7))
env_standard=800
x=range(len(hopper_record_average))
plt.plot(x,hopper_without_average,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
plt.plot(x,hopper_record_average,label='Coached by PID Controller\nEvaluation %s'%evalu_ave,color='magenta')
plt.xlabel('Episode Number', fontsize='large')
plt.ylabel('Episode Reward', fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 18})
plt.axhline(y=env_standard, color='black', linestyle='dotted')
plt.savefig('hopper.png')