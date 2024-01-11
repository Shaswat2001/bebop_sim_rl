import gym
from math import pi
from gym import spaces
import numpy as np
from .QuadrotorObsEnv import QuadrotorObsEnv

class QuadrotorTeachEnv(QuadrotorObsEnv):
    
    def __init__(self,controller = None,load_obstacle = False):
        
        super().__init__(controller,load_obstacle)

        self.max_obs_pos = np.array([6,6,6])
    
    def set_env_variable(self,n_obstacles = 0, obs_centre = None):

        self.n_obstacles = int(n_obstacles)
        self.obs_centre = obs_centre
        self.grid.reset()

        obs_centre_list = np.round(np.random.uniform(-self.obs_centre,self.obs_centre,(self.n_obstacles,3)),1)

        self.coordinate = []
        for i in range(self.n_obstacles):
            ctr_coordinate = obs_centre_list[i]

            min_max_coordinate = [ctr_coordinate - self.grid.obs_dim/2,ctr_coordinate + self.grid.obs_dim/2]
            self.coordinate.append(min_max_coordinate)
            
            print(f"Adding Obstacle : {i}")
            self.grid.add_obstacle(ctr_coordinate)
