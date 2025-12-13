#!/usr/bin/env python3
import sys
import signal
import atexit
sys.path.append("FoundationPose")
from datareader import *
from create_camera import CreateRealsense
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
import json
import threading
from datetime import datetime
import gc
import torch
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
###from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose, detect_dent_orientation
from camera_reader import CameraReader
# from level2_action import detect_object_pose_using_foundation_pose, choose_grasp_pose, execute_grasp_action, detect_object_orientation, adjust_object_orientation, detect_contact_with_surface
from env import create_env
import rospy
from std_msgs.msg import Float64MultiArray


camera = None
angle_camera = None
contact_camera = None
dobot = None
gripper = None
preview_running = None


# def _cleanup_resources():
#     """释放相机、机械臂和窗口等资源"""
#     global camera, angle_camera, contact_camera, dobot, preview_running
#     try:
#         if preview_running:
#             preview_running.clear()
#             print("[清理] 相机预览线程已停止")
#     except Exception:
#         pass
#     try:
#         if angle_camera and getattr(angle_camera, "cap", None):
#             angle_camera.cap.release()
#     except Exception:
#         pass
#     try:
#         if contact_camera and getattr(contact_camera, "cap", None):
#             contact_camera.cap.release()
#     except Exception:
#         pass
#     try:
#         if camera:
#             camera.release()
#     except Exception:
#         pass
#     try:
#         if dobot:
#             dobot.stop()
#             dobot.disable_robot()
#     except Exception:
#         pass
#     cv2.destroyAllWindows()
# def _signal_handler(signum, frame):
#     print("\n[中断] 用户终止程序")
#     _cleanup_resources()
#     try:
#         rospy.signal_shutdown("User interrupt")
#     except Exception:
#         pass
#     sys.exit(0)
# signal.signal(signal.SIGINT, _signal_handler)
# atexit.register(_cleanup_resources)

# cd serl/serl_dobot
# conda activate foundationpose
# python test_dobot.py

# cd serl/serl_dobot && conda activate foundationpose && python test_dobot.py

if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    # # 创建带时间戳的保存目录
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = os.path.join("record_images_during_grasp", timestamp)
    # os.makedirs(save_dir, exist_ok=True)
    # # print(f"图像将保存到: {save_dir}")
    # angle_log_path = os.path.join(save_dir, "angle_log.csv")
    # with open(angle_log_path, 'w') as f:
    #     f.write("frame,timestamp,angle_z_deg,detected_angles,avg_angle\n")
    # # print(f"角度数据将保存到: {angle_log_path}")

    # 使用 env.py 初始化环境（包含机械臂、相机等）
    env = create_env("config.json")
    robot_main = env.robot1
    dobot = robot_main["robot"]
    gripper = env.gripper
    # camera_main = env.camera1_main
    # camera = camera_main["cam"]  # 为清理函数设置全局变量
    # T_ee_cam = camera_main["T_ee_cam"]
    wait_init = rospy.Rate(1/2)  
    wait_init.sleep()

    gripper.control(position=600, force=10, speed=7)

    pose_now = dobot.get_pose()
    print("pose_now",pose_now)
    dobot.move_to_pose(pose_now[0]+10, pose_now[1], pose_now[2]+20, pose_now[3], pose_now[4], pose_now[5], speed=10, acceleration=3)

    camera = CameraReader(camera_id=17) #! 注意id有时会变化

    camera = CameraReader(camera_id=16) #! 注意id有时会变化



    
    



        



