#!/usr/bin/env python3

import rospy 
import pickle
from laser_assembler.srv import AssembleScans
from sensor_msgs.msg import PointCloud

rospy.init_node("assemble_scans_to_cloud")
rospy.wait_for_service("assemble_scans2")
assemble_scans = rospy.ServiceProxy('assemble_scans', AssembleScans)
pub = rospy.Publisher ("/laser_pointcloud", PointCloud, queue_size=1)

r = rospy.Rate (1)

while not rospy.is_shutdown():
    try:
        resp = assemble_scans(rospy.Time(0,0), rospy.get_rostime())
        print("Got cloud with %u points" % len(resp.cloud.points))
        pub.publish (resp.cloud)

    except rospy.ServiceException as e:
        print("Service call failed")

    point = [[point.x,point.y,point.z] for point in resp.cloud.points]
    chanel = [[channels.name,channels.values] for channels in resp.cloud.channels]

    print(point)
