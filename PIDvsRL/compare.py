from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym
import statistics 

rl=[]
pid=[]
rl_average=[]
pid_average=[]



ip_without=pickle.load(open( "ip_without_record.p", "rb"))
ip_record=pickle.load(open( "ip_record.p", "rb"))
ip_evaluation_record_without=pickle.load(open( "ip_evaluation_without_record.p", "rb"))
ip_evaluation_record=pickle.load(open( "ip_evaluation_record.p", "rb"))

double_without=pickle.load(open( "double_without_record.p", "rb"))
double_record=pickle.load(open( "double_record.p", "rb"))
double_evaluation_record_without=pickle.load(open( "double_evaluation_without_record.p", "rb"))
double_evaluation_record=pickle.load(open( "double_evaluation_record.p", "rb"))

hopper_without=pickle.load(open( "hopper_without_record.p", "rb"))
hopper_record=pickle.load(open( "hopper_record.p", "rb"))
hopper_evaluation_record_without=pickle.load(open( "hopper_evaluation_without_record.p", "rb"))
hopper_evaluation_record=pickle.load(open( "hopper_evaluation_record.p", "rb"))

walker_without=pickle.load(open( "walker_without_record.p", "rb"))
walker_record=pickle.load(open( "walker_record.p", "rb"))[2][0]
walker_evaluation_record_without=pickle.load(open( "walker_evaluation_without_record.p", "rb"))
walker_evaluation_record=pickle.load(open( "walker_evaluation_record.p", "rb"))[2][0]


n_groups = 4
standard=[800,7000,800,800]
without=[ip_without,double_without,hopper_without,walker_without]
coached=[ip_record,double_record,hopper_record,walker_record]
average_over=[20,150,100,100]

evaluation_without=[ip_evaluation_record_without,double_evaluation_record_without,hopper_evaluation_record_without,walker_evaluation_record_without]
evaluation=[ip_evaluation_record,double_evaluation_record,hopper_evaluation_record,walker_evaluation_record]


name=['ip','double','hopper','walker']

#get bounds

without_ave=[]
coached_ave=[]

without_sd=[]
coached_sd=[]

for i in range(len(name)):
    actual_without_record=without[i]
    actual_record=coached[i]
    braket_size=average_over[i]
    start_point=0

    without_average=[]
    coached_average=[]

    without_standard_deviation=[]
    coached_standard_deviation=[]

    for j in range(len(actual_record)-braket_size+1):
        braket_without=actual_without_record[start_point:start_point+braket_size]
        without_mean=statistics.mean(braket_without)
        without_average.append(without_mean)
        without_standard_deviation.append(statistics.stdev(braket_without, xbar = without_mean))

        braket_coached=actual_record[start_point:start_point+braket_size]
        coached_mean=statistics.mean(braket_coached)
        coached_average.append(coached_mean)
        coached_standard_deviation.append(statistics.stdev(braket_coached, xbar = coached_mean))

        start_point+=1

    without_sd.append(without_standard_deviation)
    coached_sd.append(coached_standard_deviation)
    without_ave.append(without_average)
    coached_ave.append(coached_average)
#plot training results
for i in range(len(name)):
    fig=plt.figure(figsize=(13,7))
    without_record=np.array(without_ave[i])
    coached_record=np.array(coached_ave[i])
    without_standard_deviation=np.array(without_sd[i])
    coached_standard_deviation=np.array(coached_sd[i])

    evalu_without=evaluation_without[i]
    evalu=evaluation[i]
    evalu_without_ave=int(sum(evalu_without)/len(evalu_without))
    evalu_ave=int(sum(evalu)/len(evalu))

    env_standard=standard[i]

    x=range(len(without_record))
    plt.plot(x,without_record,label='Normal Training\nEvaluation %s'%evalu_without_ave,color='black',linestyle='-.')
    plt.fill_between(x, without_record - without_standard_deviation, without_record+without_standard_deviation,color='gray',alpha=0.3)
    plt.plot(x,coached_record,label='Coached by PID Controller\nEvaluation %s'%evalu_ave,color='royalblue')
    plt.fill_between(x, coached_record - coached_standard_deviation, coached_record+coached_standard_deviation,color='royalblue',alpha=0.3)
    plt.xlabel('Episode Number', fontsize=25)
    plt.xticks(fontsize=18) 
    plt.ylabel('Episode Reward', fontsize=25)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 20})
    plt.axhline(y=env_standard, color='black', linestyle='dotted')
    plt.savefig('%s.png' %name[i])

for k in range(n_groups):
	for i in range(len(without_ave[k])):
		if without_ave[k][i]>=standard[k]:
			rl_average.append(i+average_over[k]-1)
			break

for k in range(n_groups):
	for i in range(len(coached_ave[k])):
		if coached_ave[k][i]>=standard[k]:
			pid_average.append(i+average_over[k]-1)
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