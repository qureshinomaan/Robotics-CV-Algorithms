#======================================================================================#
# Let's implement a PID control in one dimenstion #
#======================================================================================#

#======================================================================================#
# Defining the state parameters #
#======================================================================================#
x_target = 0.0
x_initial = -200.0
v_initial = 0.0
v_max = 5
next = lambda x, v : x+v
#======================================================================================#

#======================================================================================#
# importing important libraries #
#======================================================================================#
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

plt.style.use('fivethirtyeight')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.set_xlim([-100,100])
ax1.set_ylim([-100,100])
line = ax1.scatter( [0, 0], [x_initial, x_target], s = 50, c=['g','r'], marker='o')

ax2.set_xlim([0,200])
ax2.set_ylim([-100,100])
line2, = ax2.plot([0], [x_target - x_initial], lw=1, label = 'Error with Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Error')


def init():
    line.set_offsets([[0, x_initial], [0, x_target]])
    return line,

def init2():
    global line2
    line2.set_data([0], [x_target - x_initial])
    return line2,
#======================================================================================#



#======================================================================================#
# Defining the control Parameters #
#======================================================================================#
K_p, K_d, K_i = 0.4, 0.8, 0.4
#======================================================================================#

#======================================================================================#
# Visualising the controller #
#======================================================================================#


x_current = x_initial
v_current = v_initial
v_last, integral = 0, 0
sensor_noise, motion_noise = 0, 0
error_list = []
index =[]

def PID_Control(i):
    global x_current, v_current, v_last, integral
    global sensor_noise, motion_noise
    sensor_noise = random.gauss(0, 0.5)
    motion_noise = random.gauss(0, v_current/100)
    sensor_reading = x_current + sensor_noise
    error = x_target - sensor_reading
    v_current = K_p * error - K_d * v_last + K_i * integral
    v_last = v_current
    integral += error
    
    x_current = x_current + v_current - 3 + motion_noise
    print(error, x_current, v_current)
    line.set_offsets([[0, x_current], [0, x_target]])
    
    return line,
    
def PID_Graph(i):
    global x_current, v_current, v_last, integral
    global sensor_noise, motion_noise
    global error_list
    sensor_reading = x_current + sensor_noise
    error = x_target - x_current
    error_list.append(error)
    index.append(i*3)
    line2.set_data(index, error_list)
    return (line2,)
    
anim = animation.FuncAnimation(fig, PID_Control,
                            init_func = init,
                            frames = 500000,
                            interval = 100)
                            
anim2 = animation.FuncAnimation(fig, PID_Graph,
                                init_func= init2,
                                frames = 500000,
                                interval = 100)
plt.show()
    

    
#======================================================================================#
