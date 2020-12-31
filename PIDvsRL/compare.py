from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

rl=[]
pid=[]


average_over_ip=10
average_over=50

ip_without=pickle.load(open( "ip_without_record.p", "rb"))
ip_record_temp=pickle.load(open( "ip_record.p", "rb"))
ip_record=ip_record_temp[0][0]
double_without=pickle.load(open( "double_without_record.p", "rb"))
double_record_temp=pickle.load(open( "double_record.p", "rb"))
double_record=double_record_temp[0][0]
hopper_without=pickle.load(open( "hopper_without_record.p", "rb"))
hopper_record_temp=pickle.load(open( "hopper_record.p", "rb"))
hopper_record=hopper_record_temp[0][0]

n_groups = 3
standard=[800,5500,800]
without=[ip_without,double_without,hopper_without]
coached=[ip_record,double_record,hopper_record]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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

labels = ['Inverted\nPendulum', 'Double\nInverted\nPendulum', 'Hopper']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pid, width, label='With PID Coaching')
rects2 = ax.bar(x + width/2, rl, width, label='Without Coaching')

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

fig.tight_layout()

plt.savefig('compare.png')
