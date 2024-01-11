#!/usr/bin/env python3

from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np

class GazeboSubscriber(Node):

    def __init__(self):
        super().__init__('gazebo_subscriber')

        self.rgb_data = np.zeros((12,64,64))
        self.depth_data = np.zeros((4,64,64))
        self.rgb_index = 0
        self.depth_index = 0
        self.bridge = CvBridge()
        self.rgb_subscription = self.create_subscription(Image,"/camera_link_camera/image_raw",self.rgb_callback,10)
        self.depth_subscription = self.create_subscription(Image,"/camera_link_camera/depth/image_raw",self.depth_callback,10)
        self.rgb_subscription  # prevent unused variable warning
        self.depth_subscription

    def get_data(self):

        self.rgb_index = 0
        self.depth_index = 0

        while self.rgb_index < 4:
            print(self.rgb_index)
            continue

        return self.rgb_data,self.depth_data

    def rgb_callback(self, msg):
        print("I am IN")
        current_rbg_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        index = self.rgb_index % 4
        self.rgb_data[3*index:3*(index+1),:,:] = current_rbg_frame.T
        self.rgb_index +=1

    def depth_callback(self, msg):
        print("I am in")
        current_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        index = self.depth_index % 4
        self.depth_data[index,:,:] = current_depth_frame.T
        self.depth_index +=1

if __name__ == "__main__":

    rclpy.init(args=None)

    minimal_subscriber = GazeboSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


