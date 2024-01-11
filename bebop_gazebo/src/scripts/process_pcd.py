#!/usr/bin/env python3

from sensor_msgs.msg import PointCloud2
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np
import ros2_numpy

class GazeboSubscriber(Node):

    def __init__(self):
        super().__init__('gazebo_subscriber')

        self.rgb_data = np.zeros((12,64,64))
        self.depth_data = np.zeros((4,64,64))
        self.rgb_index = 0
        self.depth_index = 0
        self.bridge = CvBridge()
        self.rgb_subscription = self.create_subscription(PointCloud2,"/rtabmap/cloud_map",self.rgb_callback,10)
        self.rgb_subscription  # prevent unused variable warning

    def rgb_callback(self, msg):
        print("I am IN")
        data = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        print(max(data[:,1]))

if __name__ == "__main__":

    rclpy.init(args=None)

    minimal_subscriber = GazeboSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()