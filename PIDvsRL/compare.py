from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

rl=[]
pid=[]
rl_average=[]
pid_average=[]


average_over=20

ip_without=pickle.load(open( "ip_without_record.p", "rb"))
ip_record_temp=pickle.load(open( "ip_record.p", "rb"))
ip_record=ip_record_temp[0][0]
double_without=pickle.load(open( "double_without_record.p", "rb"))
double_record_temp=pickle.load(open( "double_record.p", "rb"))
double_record=double_record_temp[0][0]
hopper_without=pickle.load(open( "hopper_without_record.p", "rb"))
hopper_record_temp=pickle.load(open( "hopper_record.p", "rb"))
hopper_record=hopper_record_temp[0][0]
walker_without=pickle.load(open( "walker_without_record.p", "rb"))
walker_record_temp=pickle.load(open( "walker_record.p", "rb"))
walker_record=walker_record_temp[2][0]


n_groups = 4
standard=[800,5500,800,800]
without=[ip_without,double_without,hopper_without,walker_without]
coached=[ip_record,double_record,hopper_record,walker_record]

ip_evaluation_record_without=pickle.load(open( "ip_evaluation_without_record.p", "rb"))
ip_evaluation_record=pickle.load(open( "ip_evaluation_record.p", "rb"))
ip_evaluation_record=ip_evaluation_record[0][0]
double_evaluation_record_without=pickle.load(open( "double_evaluation_without_record.p", "rb"))
double_evaluation_record=pickle.load(open( "double_evaluation_record.p", "rb"))
double_evaluation_record=double_evaluation_record[0][0]
hopper_evaluation_record_without=pickle.load(open( "hopper_evaluation_without_record.p", "rb"))
hopper_evaluation_record=pickle.load(open( "hopper_evaluation_record.p", "rb"))
hopper_evaluation_record=hopper_evaluation_record[0][0]
walker_evaluation_record_without=pickle.load(open( "walker_evaluation_without_record.p", "rb"))
walker_evaluation_record=pickle.load(open( "walker_evaluation_record.p", "rb"))
walker_evaluation_record=walker_evaluation_record[2][0]


evaluation_without=[ip_evaluation_record_without,double_evaluation_record_without,hopper_evaluation_record_without,walker_evaluation_record_without]
evaluation=[ip_evaluation_record,double_evaluation_record,hopper_evaluation_record,walker_evaluation_record]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
ip_without_average=moving_average(ip_without,average_over)
ip_record_average=moving_average(ip_record,average_over)
double_without_average=moving_average(double_without,average_over)
double_record_average=moving_average(double_record,average_over)
hopper_without_average=moving_average(hopper_without,average_over)
hopper_record_average=moving_average(hopper_record,average_over)
walker_without_average=moving_average(walker_without,average_over)
walker_record_average=moving_average(walker_record,average_over)

without_average=[ip_without_average,double_without_average,hopper_without_average,walker_without_average]
coached_average=[ip_record_average,double_record_average,hopper_record_average,walker_record_average]

name=['ip','double','hopper','walker']

#plot training results
for i in range(len(name)):
    fig=plt.figure(figsize=(13,7))
    without_record=without_average[i]
    coached_record=coached_average[i]
    evalu_without=evaluation_without[i]
    evalu=evaluation[i]
    evalu_without_ave=int(sum(evalu_without)/len(evalu_without))
    evalu_ave=int(sum(evalu)/len(evalu))
    env_standard=standard[i]
    x=range(len(without_record))
    plt.plot(x,without_record,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
    plt.plot(x,coached_record,label='With PID Controller as Coach\nEvaluation %s'%evalu_ave,color='magenta')
    plt.xlabel('Episode Number', fontsize='large')
    plt.ylabel('Episode Reward', fontsize='large')
    plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 10})
    plt.axhline(y=env_standard, color='black', linestyle='dotted')
    plt.savefig('%s.png' %name[i])

for k in range(n_groups):
	for i in range(len(without_average[k])):
		if without_average[k][i]>=standard[k]:
			rl_average.append(i+average_over-1)
			break
for k in range(n_groups):
	for i in range(len(coached_average[k])):
		if coached_average[k][i]>=standard[k]:
			pid_average.append(i+average_over-1)
			break

for k in range(n_groups):
	count=0
	first_time=0
	index=0
	total=5
	for i in range(len(without[k])):
		if without[k][i]>=standard[k]:
			if first_time==0:
				count=1
				index=i
				first_time=1
				total-=1
			elif i-index==1:
				count+=1
				index=i
				total-=1
				if total==0:
					rl.append(index)
					break
			else:
				count=1
				total=4
				index=i

for k in range(n_groups):
	count=0
	first_time=0
	index=0
	total=5
	for i in range(len(coached[k])):
		if coached[k][i]>=standard[k]:
			if first_time==0:
				count=1
				index=i
				first_time=1
				total-=1
			elif i-index==1:
				count+=1
				index=i
				total-=1
				if total==0:
					pid.append(index)
					break
			else:
				count=1
				total=4
				index=i

# create plot

print('rl:',rl)
print('rl_average',rl_average)
print('pid:',pid)
print('pid_average',pid_average)


labels = ['Inverted\nPendulum', 'Double\nInverted\nPendulum', 'Hopper','Walker']
x = np.arange(len(labels))  # the label locations
width = 0.35/2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*3/2, pid, width, label='With PID Coaching\n 5 Consecutive Wins')
rects3 = ax.bar(x - width/2, pid_average, width, label='With PID Coaching\n Average over 20')
rects2 = ax.bar(x + width/2, rl, width, label='Without Coaching\n 5 Consecutive Wins')
rects4 = ax.bar(x + width*3/2, rl_average, width, label='Without Coaching\n Average over 20')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Episode Number')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.savefig('compare.png')