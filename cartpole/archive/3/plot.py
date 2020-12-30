import pickle
import matplotlib.pyplot as plt
import numpy as np

#read data
reward_record=pickle.load(open( "cartpolerecord.p", "rb"))
reward_record_without=pickle.load(open( "cartpole_without_record.p", "rb"))

#define the length of the vector
episode_number=100
average_over=10
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

prohibition_parameter=[-15,-20,-25,-30]
prohibition_position=[0.9,0.95]

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
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('cartpole_with_angle_boundary_at_%s_plot.png' %prohibition_position[i])
