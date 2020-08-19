import numpy as np
from interaction_matrix import interactionMatrix 
from final import update_v
from final import sim_settings, make_cfg
import habitat_sim
import random
from run_a_pair import run_a_pair
import cv2


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def get_vik():
    nx, ny = (512,384)
    vik=np.array([])
    for i in range(nx):
        for j in range(ny):
            vik=np.hstack((vik,np.array([i,j])))
            
    return vik

def controller(V, vik, filename, observations):
    nx, ny = (512,384)
    error = readFlow(filename)
    error_i = error
    error=error.transpose(1,0,2)
    error=error.flatten()
    error=np.reshape(error,(nx*ny*2,-1))
    flow_depth=np.linalg.norm(error_i,axis=2)
    flow_depth = observations["depth_sensor"]
    flow_depth=flow_depth.astype('float64')
    print(flow_depth.shape)
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    cam=np.asarray([[nx/2,0,nx/2],[0,ny/2, ny/2 ],[ 0, 0, 1]])
    print("Error : ", np.sum(error))
    Lsd=interactionMatrix(vik,cam,flow_depth)
    lamda=0.01
    mu=0.03
    H=np.matmul(Lsd.T,Lsd)
    V=-lamda*np.matmul(np.matmul(np.linalg.pinv(H+mu*H.diagonal()),Lsd.T),error)
    print("Velocity Commands: ", V)
    return V

def take_action(sim, V):
    sim.step("move_left")
    sim.step("move_right")
    sim.step("move_forward")
    sim.step("move_backward")
    sim.step("move_up")
    sim.step("move_down")
    sim.step("look_up")
    sim.step("look_down")
    sim.step("look_left")
    sim.step("look_right")
    sim.step("look_clock")
    observations = sim.step("look_anti")
    return observations, sim


ref_image = './ref_image.png'
cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
vik = get_vik()
V = np.array([0, 0, 0, 0, 0, 0])
iter = 0

while iter < 1500:
    print("iter :", iter)
    iter = iter + 1
    observations, sim = take_action(sim, V)
    cv2.imwrite('observations/color'+str(iter)+'.png', observations["color_sensor"])
    run_a_pair(ref_image, 'observations/color'+str(iter)+'.png', iter)
    V = controller(V, vik, 'test.flo', observations)
    sim = update_v(V, sim)
    
    
