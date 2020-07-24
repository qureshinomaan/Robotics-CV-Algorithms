#======================================================================================#
# Let's implement a PID control in one dimenstion #
#======================================================================================#

#======================================================================================#
# Defining the state parameters #
#======================================================================================#
x_target = 0.0
x_initial = 100.0
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
  
line = axis.scatter([x_initial, x_target], [0, 0], s = 50, c=['g','r'], marker='o')
  

def init():
    line.set_offsets([[x_initial, 0], [x_target, 0]])
    return line,
  
#======================================================================================#



#======================================================================================#
# Defining the control Parameters #
#======================================================================================#
K_p, K_d, K_i = 1, 1, 1
#======================================================================================#

#======================================================================================#
# Visualising the controller #
#======================================================================================#


x_current = x_initial
v_current = v_initial

def K_Control(i):
    global x_current, v_current
    sensor_reading = x_current + random.gauss(0, 0.5)
    error = x_target - sensor_reading
    if error < 0.3 and error > 0.3 :
        v_current = 0
    elif error > 0 :
        v_current = min(v_max, K_p * error) + random.gauss(0, min(v_max, K_p * error)/10)
    else :
        v_current = max(-v_max, K_p * error) + random.gauss(0, max(-v_max, K_p * error)/10)
    x_current = x_current + v_current
    print(error, x_current, v_current)
    line.set_offsets([[x_current, 0], [x_target, 0]])
    
    return line,
    
anim = animation.FuncAnimation(fig, K_Control,
                            init_func = init,
                            frames = 500000,
                            interval = 100)
plt.show()
    

    
#======================================================================================#
