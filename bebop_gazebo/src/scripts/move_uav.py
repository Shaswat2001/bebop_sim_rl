#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from sensor_msgs.msg import Image
from math import pi
import numpy as np
import time

class GazeboPublisher(Node):

    def __init__(self):
        super().__init__('gazebo_publisher')
        self.uav_publisher = self.create_publisher(String, '/drone_body_string', 10)
        self.manip_publisher = self.create_publisher(String, '/robot_joints_string', 10)

    def publish_uav(self,pose):
        msg = String()
        msg.data = pose
        self.uav_publisher.publish(msg)

    def publish_manip(self,joint_val):
        msg = String()
        msg.data = joint_val
        self.manip_publisher.publish(msg)

    def publish_robot(self,pose,joint_val):
        msg = String()
        msg.data = pose
        self.uav_publisher.publish(msg)

        msg = String()
        msg.data = joint_val
        self.manip_publisher.publish(msg)

if __name__ == "__main__":

    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    publisher = GazeboPublisher()

    rate = node.create_rate(0.01)
    init_height = 0.5
    for i in range(10):
        height = init_height + 0.01*i
        manip_joint = f"0,0,{height},0,0,0"
        publisher.publish_uav(manip_joint)

        time.sleep(1)

    for i in range(200):
        angle = 0.0314*i
        manip_joint = f"0,0,{height},0,0,{angle}"
        publisher.publish_uav(manip_joint)

        time.sleep(1)

    init_height = height
    for i in range(10):
        height = init_height + 0.01*i
        manip_joint = f"0,0,{height},0,0,{angle}"
        publisher.publish_uav(manip_joint)

        time.sleep(1)

    init_angle = angle
    for i in range(200):
        angle = init_angle - 0.0314*i
        manip_joint = f"0,0,{height},0,0,{angle}"
        publisher.publish_uav(manip_joint)

        time.sleep(1)

