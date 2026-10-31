#!/usr/bin/env python3
"""grab_frame.py - save one camera frame from the running simulation (to look at what the robot sees).
   python3 $D/grab_frame.py [output.png]      (default: /tmp/diffdrivebot/gazebo_frame.png)"""
import sys
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/diffdrivebot/gazebo_frame.png'
rclpy.init()
node = Node('grab_frame')
done = []


def cb(msg):
    cv2.imwrite(out, CvBridge().imgmsg_to_cv2(msg, 'bgr8'))
    done.append(1)


node.create_subscription(Image, '/camera/image_raw', cb, 1)
for _ in range(30):
    rclpy.spin_once(node, timeout_sec=1.0)
    if done:
        break
print('saved ' + out if done else 'no image received on /camera/image_raw (is the simulation running?)')
rclpy.shutdown()
