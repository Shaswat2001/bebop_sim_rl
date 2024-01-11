import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import time
from uam_msgs.srv import ResponseUavPose
from uam_msgs.srv import RequestUavPose
from uam_msgs.srv import RequestUavVel
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState

from std_srvs.srv import Empty

class UavClientAsync(Node):

    def __init__(self):
        super().__init__('uam_client_async')
        self.cli = self.create_client(RequestUavVel, 'get_uav_vel')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestUavVel.Request()

    def send_request(self, uav_vel):
        self.req.uav_vel = uav_vel
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

# class CollisionSubscriber(Node):

#     def __init__(self):
#         super().__init__('collision_subscriber')

#         self.contact_bumper_msg = None
#         self.collision = False
#         self.collision_subscription = self.create_subscription(ContactsState,"/bumper_states",self.collision_callback,10)
#         self.collision_subscription  # prevent unused variable warning

#     def get_collision_info(self):
        
#         if self.collision:
#             self.collision = False
#             return True 
        
#         return False

#     def collision_callback(self, msg):
#         self.contact_bumper_msg = msg

#         if self.contact_bumper_msg is not None and len(self.contact_bumper_msg.states) != 0:
#             self.collision = True

class ResetSimClientAsync(Node):

    def __init__(self):
        super().__init__('reset_sim_client_async')
        self.cli = self.create_client(RequestUavPose, 'get_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestUavPose.Request()

    def send_request(self,pose_string = "0, 0, 2, 0, 0, 0"):
        uav_msg = String()
        uav_msg.data = pose_string
        self.req.uav_pose = uav_msg
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

class GetUavPoseClientAsync(Node):

    def __init__(self):
        super().__init__('get_uav_pose_client_async')
        self.cli = self.create_client(ResponseUavPose, 'send_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ResponseUavPose.Request()

    def send_request(self):
        self.req.get_pose = 1
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        pose_uav = self.future.result().pose_uav.position
        return np.array([pose_uav.x,pose_uav.y,pose_uav.z])

class PauseGazeboClient(Node):

    def __init__(self):
        super().__init__('pause_gazebo_client')
        
        self.cli = self.create_client(Empty, '/pause_physics')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def send_request(self):
        self.cli.call(Empty.Request())

class UnPauseGazeboClient(Node):

    def __init__(self):
        super().__init__('unpause_gazebo_client')
        
        self.cli = self.create_client(Empty, '/unpause_physics')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def send_request(self):
        self.cli.call(Empty.Request())

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        self.lidar_range = None
        self.ee_pose_subscription = self.create_subscription(LaserScan,"/laser_controller/out",self.lidar_callback,10)
        self.ee_pose_subscription  # prevent unused variable warning

    def get_state(self):

        if self.lidar_range is None:
            return np.zeros(shape=(360)),False
        
        lidar_data = np.array(self.lidar_range)
        contact = False
        for i in range(lidar_data.shape[0]):
            if lidar_data[i] == np.inf:
                lidar_data[i] = 1
            elif lidar_data[i] < 0.2:
                contact = True

        return lidar_data,contact

    def lidar_callback(self, msg):

        self.lidar_range = msg.ranges

class BaseGazeboUAVVelObsEnv(gym.Env):
    
    def __init__(self,controller = None): 
        
        self.uam_publisher = UavClientAsync()
        self.get_uav_pose_client = GetUavPoseClientAsync()
        self.pause_sim = PauseGazeboClient()
        self.unpause_sim = UnPauseGazeboClient()
        self.lidar_subscriber = LidarSubscriber()
        self.reset_sim = ResetSimClientAsync()
        # self.collision_sub = CollisionSubscriber()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.get_uav_pose_client)
        self.executor.add_node(self.pause_sim)
        self.executor.add_node(self.unpause_sim)
        self.executor.add_node(self.lidar_subscriber)
        self.executor.add_node(self.reset_sim)
        # self.executor.add_node(self.collision_sub)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 363
        self.action_max = np.array([0.045,0.045,0.045])
        
        self.q = None
        self.qdot = None
        self.q_des = None
        self.qdot_des = None
        self.qdotdot_des = None
        self.man_pos = None
        self.manip_difference = None

        self.max_time = 55
        self.dt = 0.02
        self.current_time = 0

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([1.5,1.5,1.5])
        self.min_q_bound = np.array([-1.5,-1.5,-1.5])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        action = action[0]
        self.vel = self.vel + action[:3]

        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)

        self.publish_simulator(self.vel)

        lidar,self.check_contact = self.get_lidar_data()
        self.get_uav_pose()
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pose_diff = self.q_des - self.pose
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        if self.const_broken:

            self.get_safe_pose()
            uav_pos_ort = list(self.previous_pose)[0:3]
            uav_pos_ort += [0,0,0]
            uav_pos_ort = f"{uav_pos_ort}"[1:-1]

            self.reset_sim.send_request(uav_pos_ort)

            self.vel = self.vel - action[:3]
            # self.publish_simulator(self.vel)

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error

        if not self.const_broken:
            self.previous_pose = self.pose
            # if pose_error < 0.01:
            #     done = True
            #     reward = 1000
            # elif pose_error < 0.05:
            #     done = True
            #     reward = 100
            if pose_error < 0.1:
                done = True
                reward = 10
            # elif pose_error < 1:
            #     done = True
            #     reward = 0
            else:
                reward = -(pose_error*5)
        
        else:
            reward = -20
            # done = True

        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.vel.shape[0]):
                if self.vel[i] > self.max_q_bound[i]:
                    constraint+= (self.vel[i] - self.max_q_bound[i])*10
                elif self.vel[i] < self.min_q_bound[i]:
                    constraint+= abs(self.vel[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.vel.shape[0]):
                constraint+= (abs(self.vel[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.vel > self.max_q_safety) or np.any(self.vel < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        
        if self.check_contact:
            return True
        
        # if np.any(self.vel[:3] > self.max_q_bound[:3]) or np.any(self.vel[:3] < self.min_q_bound[:3]):
        #     return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.pose - self.q_des) 

        return pose_error
        
    def reset(self):

        #initial conditions
        self.pose = np.array([0,0,2])
        self.vel = np.array([0,0,0])
        self.previous_pose = np.array([0,0,2])
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.vel)
        self.reset_sim.send_request()
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        self.max_time = 55
        time.sleep(0.1)

        return prp_state
    
    def reset_test(self,q_des,max_time):

        #initial conditions
        self.pose = np.array([0,0,2])
        self.vel = np.array([0,0,0])
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        self.q_des = q_des
        self.max_time = max_time
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.vel)
        self.reset_sim.send_request()
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        time.sleep(0.1)

        return prp_state
    
    def publish_simulator(self,q):
    
        uav_vel = list(q)[0:3]
        uav_vel = f"{uav_vel}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_vel

        self.uam_publisher.send_request(uav_msg)

        self.unpause_sim.send_request()

        time.sleep(self.dt)

        self.pause_sim.send_request()

    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact

    def get_uav_pose(self):
    
        self.pose = self.get_uav_pose_client.send_request()   
    
    def get_safe_pose(self):

        for i in range(len(self.previous_pose) - 1):

            if self.previous_pose[i] < 0:
                self.previous_pose[i]+= 0.05
            else:
                self.previous_pose[i]-= 0.05

if __name__ == "__main__":

    rclpy.init()

    env = BaseGazeboUAVVelObsEnv()
    env.reset()

    action = np.array([0,0,0,0,0,0,0]).reshape(1,-1)

    prp_state, reward, done, info = env.step(action)

    print(env.q)
    print(info["constraint"])