#!/usr/bin/env python3

import rospy
import pickle
from geometry_msgs.msg import Twist

f = open("/home/shaswat/ros_ws/src/bebop_simulator/bebop_gazebo/src/velocity.pkl","rb")
vel_val = pickle.load(f)
f.close()

print(vel_val)

def init():
    rospy.init_node('test_tl')
    twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    for i in vel_val:
        vel = Twist()
        vel.linear.x = i[0]
        vel.linear.y = i[1]
        vel.linear.z = i[2]
        twist_pub.publish(vel)
        rospy.sleep(0.15)

if __name__ == '__main__':
    init()


