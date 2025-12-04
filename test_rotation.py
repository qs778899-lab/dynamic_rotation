#!/usr/bin/env python3
import sys
import signal
import atexit
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD 
from qwen_mask import get_mask_from_qwen
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
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
###from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose, detect_dent_orientation
from camera_reader import CameraReader
from level2_action import detect_object_pose_using_foundation_pose, choose_grasp_pose, execute_grasp_action, detect_object_orientation, adjust_object_orientation, detect_contact_with_surface
from env import create_env
import rospy
from std_msgs.msg import Float64MultiArray


camera = None
angle_camera = None
contact_camera = None
dobot = None
gripper = None
preview_running = None


def _cleanup_resources():
    """释放相机、机械臂和窗口等资源"""
    global camera, angle_camera, contact_camera, dobot, preview_running
    try:
        if preview_running:
            preview_running.clear()
            print("[清理] 相机预览线程已停止")
    except Exception:
        pass
    try:
        if angle_camera and getattr(angle_camera, "cap", None):
            angle_camera.cap.release()
    except Exception:
        pass
    try:
        if contact_camera and getattr(contact_camera, "cap", None):
            contact_camera.cap.release()
    except Exception:
        pass
    try:
        if camera:
            camera.release()
    except Exception:
        pass
    try:
        if dobot:
            dobot.stop()
            dobot.disable_robot()
    except Exception:
        pass
    cv2.destroyAllWindows()
def _signal_handler(signum, frame):
    print("\n[中断] 用户终止程序")
    _cleanup_resources()
    try:
        rospy.signal_shutdown("User interrupt")
    except Exception:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_cleanup_resources)



if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"图像将保存到: {save_dir}")
    angle_log_path = os.path.join(save_dir, "angle_log.csv")
    with open(angle_log_path, 'w') as f:
        f.write("frame,timestamp,angle_z_deg,detected_angles,avg_angle\n")
    # print(f"角度数据将保存到: {angle_log_path}")

    # 使用 env.py 初始化环境（包含机械臂、相机等）
    env = create_env("config.json")
    robot_main = env.robot1
    dobot = robot_main["robot"]
    gripper = env.gripper
    camera_main = env.camera1_main
    camera = camera_main["cam"]  # 为清理函数设置全局变量
    T_ee_cam = camera_main["T_ee_cam"]
    
    # 从 GraspLibrary.json 加载抓取参数
    target_object = "stirring rod"  # 可以修改为: "red cylinder", "red stirring rod", "stirring rod"
    with open("GraspLibrary.json", 'r') as f:
        grasp_library = json.load(f)
    if target_object not in grasp_library:
        raise ValueError(f"目标物体 '{target_object}' 不在 GraspLibrary.json 中")
    grasp_params = grasp_library[target_object]
    print(f"\n加载目标物体: {target_object}")
    print(f"抓取参数: {grasp_params}\n")
    
    # mesh_file = "mesh/cube.obj"
    # mesh_file = "mesh/thin_cube.obj"
    mesh_file = "mesh/cube_1_20.obj"
    key = cv2.waitKey(1)
 
    # 从 GraspLibrary 获取抓取参数
    z_xoy_angle = grasp_params["z_xoy_angle"]
    vertical_euler = grasp_params["vertical_euler"]
    grasp_tilt_angle = grasp_params["grasp_tilt_angle"]
    angle_threshold = grasp_params["angle_threshold"]
    T_safe_distance = grasp_params["T_safe_distance"]
    z_safe_distance = grasp_params["z_safe_distance"]
    gripper_close_pos = grasp_params["gripper_close_pos"]

    
    
