import gym
from math import pi
from gym import spaces
import numpy as np
import random

class BaseQuadrotorEnv(gym.Env):
    
    def __init__(self,controller = None): 
        
        if controller is None:
            print("ERROR : CONTROLLER NOT DEFINED")
            return None,None,None,None
        
        self.controller = controller()
        self.state = None
        self.state_size = 12
        self.action_max = np.array([1,1,1,1,1,1,0.1,0.1,0.1]*3)
        self.safe_action_max = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.05,0.05,0.05]*3)

        self.max_angle = np.array([np.pi/6,np.pi/6,2*np.pi])

        self.pos = None
        self.lnr_vel = None
        self.ang = None
        self.ang_vel = None

        self.max_time = 15
        self.dt = 0.01
        self.current_time = 0

        self.safety_engage = np.array([5.5,5.5,5.5])
        self.pos_bound = np.array([7,7,7])
        self.vel_bound = np.array([3,3,3])
        self.ang_bound = pi/12
        self.ang_vel_bound = np.array([1.5,1.5,1.5])
        # self.observation_space = spaces.Box(-4*self.Quadrotor.minT,4*self.Quadrotor.minT, dtype=np.float64)
        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        reward = 0
        done = False 
        # position, velocity, orientation and angular velocity needed given by RL
        new_pos = self.pos + action[:3]
        new_vel = self.lnr_vel +  action[3:6]
        new_ang_vel = self.ang_vel + action[6:9]

        if np.any(abs(new_vel) > self.vel_bound):
            reward -= 100
            new_vel = np.clip(new_vel,-self.vel_bound,self.vel_bound)

        if np.any(abs(new_pos) > self.pos_bound):
            reward -= 100
            new_pos = np.clip(new_pos,-self.pos_bound,self.pos_bound)

        elif np.any(abs(new_pos) > self.safety_engage):
            reward -= 50
        
        if np.any(abs(new_ang_vel) > self.ang_vel_bound):
            reward -= 100
            new_ang_vel = np.clip(new_ang_vel,-self.ang_vel_bound,self.ang_vel_bound)
        
        self.pos,self.ang,self.lnr_vel,self.ang_vel = self.controller.move_robot(new_pos,new_vel,new_ang_vel)

        pose_error =  np.linalg.norm(self.pos - self.pos_des) 
        ang_error = np.linalg.norm(self.ang - self.ort_des)
        vel_error = np.linalg.norm(self.lnr_vel - self.lnr_vel_des)

        reward -= (pose_error*10 + ang_error*0.01 + vel_error*0.01)
        # reward -= 1

        if pose_error < 0.01:
            done = True
            reward = 1000
        elif pose_error < 0.05:
            done = True
            reward = 100
        elif pose_error < 0.1:
            done = True
            reward = 10
        elif pose_error < 1:
            done = True
            reward = 0
        elif self.current_time > self.max_time:
            done = True
            reward -= 2

        if done:
            print(f"The position error at the end : {pose_error}")
            print(f"The orientation error at the end : {ang_error}")
            print(f"The velocity error at the end : {vel_error}")
            print(f"The end pose is : {self.pos}")

        constraint = 0   
        for i in range(new_pos.shape[0]):
            if abs(new_pos[i]) > self.pos_bound[i]:
                constraint+= (abs(new_pos[i]) - self.pos_bound[i])*10

        if constraint == 0:
            for i in range(new_pos.shape[0]):
                constraint+= (abs(new_pos[i]) - self.pos_bound[i])*10

        # print(constraint)
        self.info["constraint"] = constraint
        self.info["safe_reward"] = -constraint
        self.info["safe_cost"] = 0
        self.info["negative_safe_cost"] = 0
        self.info["engage_reward"] = -10

        if np.any(np.abs(new_pos) > self.safety_engage):
            self.info["engage_reward"] = 10
            
        if constraint > 0:
            self.info["safe_cost"] = 1
            self.info["negative_safe_cost"] = -1

        self.state = np.concatenate((self.pos,self.lnr_vel,self.pos_des - self.pos,self.lnr_vel_des - self.lnr_vel))

        self.current_time += 1
        return self.state, reward, done, self.info
        
    def reset(self):

        #initial conditions
        self.pos = np.array([0., 0., 0.]) # starting location [x, y, z] in inertial frame - meters
        self.lnr_vel = np.array([0., 0., 0.]) #initial velocity [x; y; z] in inertial frame - m/s
        self.ang = np.array([0., 0., 0.]) #initial Euler angles [phi, theta, psi] relative to inertial frame in deg

        deviation = 10 # magnitude of initial perturbation in deg/s
        random_set = np.array([random.random(), random.random(), random.random()])
        self.ang_vel = np.deg2rad(2* deviation * random_set - deviation) #initial angular velocity [phi_dot, theta_dot, psi_dot]

        self.pos_des = np.random.randint(-4,5,size=(3,))
        self.ort_des = np.zeros(self.ang.shape)
        self.lnr_vel_des = np.zeros(self.lnr_vel.shape)
        self.ang_vel_des = np.zeros(self.ang_vel.shape)
        print(f"The target pose is : {self.pos_des}")
        # Add initial random roll, pitch, and yaw rates
        
        self.controller.reset(self.pos,self.lnr_vel,self.ang,self.ang_vel)

        self.state = np.concatenate((self.pos,self.lnr_vel,self.pos_des - self.pos,self.lnr_vel_des - self.lnr_vel))
        self.current_time = 0
        self.info = {}

        return self.state
     