#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import numpy as np


def tracking_data_callback(msg):
    """Callback function to process received tracking data."""
    if len(msg.data) >= 2:
        angle_z_deg = msg.data[0]
        b = msg.data[1]
        rospy.loginfo(f"Received tracking data - angle_z_deg: {angle_z_deg:.6f}, b: {b:.6f}")
        

        return angle_z_deg, b
        # Process the received data here
        # You can add your custom logic to handle angle_z_deg and b values
        
    else:
        rospy.logwarn("Received message with insufficient data")





def compute_pose(angle_z_deg, b):
    """Compute the pose of tool the gripper."""
    # rotate z axis with angle_z_deg
    # then move in y axis with b
    rotation = R.from_euler('z', -angle_z_deg, degrees=True)
    translation = np.array([0, b, 0])
    
    



    

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('tracking_data_listener', anonymous=True)
    
    # Start listening to the topic
    rospy.Subscriber("/tracking_data", Float64MultiArray, tracking_data_callback)
    
    rospy.loginfo("Listening for tracking data...")
    
    # Keep the node running
    rospy.spin()
