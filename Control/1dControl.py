#======================================================================================#
# Let's implement a PID control in one dimenstion #
#======================================================================================#

#======================================================================================#
# Defining the state parameters #
#======================================================================================#
x_target = 0.0
x_initial = -100.0
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

fig = plt.figure()
axis = plt.axes(xlim =(-100, 100),
                ylim =(-100, 100))
  
line = axis.scatter( [0, 0], [x_initial, x_target], s = 50, c=['g','r'], marker='o')
  

def init():
    line.set_offsets([[0, x_initial], [0, x_target]])
    return line,
  
#======================================================================================#



#======================================================================================#
# Defining the control Parameters #
#======================================================================================#
K_p, K_d, K_i = 0.5, 0.2, 0.01
#======================================================================================#

#======================================================================================#
# Visualising the controller #
#======================================================================================#


x_current = x_initial
v_current = v_initial
v_last, integral = 0, 0
def K_Control(i):
    global x_current, v_current, v_last, integral
    sensor_reading = x_current + random.gauss(0, 0.5)
    error = x_target - sensor_reading
    v_current = K_p * error - K_d * v_last + K_i * integral
    v_last = v_current
    integral += error
    
    x_current = x_current + v_current - 3 + random.gauss(0, v_current/100)
    print(error, x_current, v_current)
    line.set_offsets([[0, x_current], [0, x_target]])
    
    return line,
    
anim = animation.FuncAnimation(fig, K_Control,
                            init_func = init,
                            frames = 500000,
                            interval = 1000)
plt.show()
    

    
#======================================================================================#
