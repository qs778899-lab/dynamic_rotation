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
from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose, detect_dent_orientation
from camera_reader import CameraReader
from level2_action import detect_object_pose_using_foundation_pose, choose_grasp_pose, execute_grasp_action, detect_object_orientation, adjust_object_orientation
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




# 初始化函数在 env.py 中


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
    # 调用封装好的函数检测物体位姿
    center_pose = detect_object_pose_using_foundation_pose(
        target=target_object,
        mesh_path=mesh_file,
        cam=camera_main  # 使用 env 初始化的 camera_main，包含 cam 和 cam_k
    )


    key = cv2.waitKey(1)
    #! 怎么检查id是否对应
    angle_camera = CameraReader(camera_id=10, init_camera=True)   #! 用于角度检测的USB相机 
    contact_camera = CameraReader(camera_id=11, init_camera=True) #! 用于触碰检测的USB相机 


    # 将center_pose转换为numpy数组
    center_pose_array = np.array(center_pose, dtype=float)
    # 从 GraspLibrary 获取抓取参数
    z_xoy_angle = grasp_params["z_xoy_angle"]
    vertical_euler = grasp_params["vertical_euler"]
    grasp_tilt_angle = grasp_params["grasp_tilt_angle"]
    angle_threshold = grasp_params["angle_threshold"]
    T_safe_distance = grasp_params["T_safe_distance"]
    z_safe_distance = grasp_params["z_safe_distance"]
    gripper_close_pos = grasp_params["gripper_close_pos"]
    
    # 计算抓取姿态
    pre_grasp_pose, grasp_pose, T_base_ee_ideal = choose_grasp_pose(
        center_pose_array=center_pose_array,
        dobot=dobot,
        T_ee_cam=T_ee_cam,
        z_xoy_angle=z_xoy_angle,
        vertical_euler=vertical_euler,
        grasp_tilt_angle=grasp_tilt_angle,
        angle_threshold=angle_threshold,
        T_tcp_ee_z=-0.16,
        T_safe_distance=T_safe_distance,
        z_safe_distance=z_safe_distance,
        verbose=True
    )
    
    # 执行抓取动作
    success = execute_grasp_action(
        grasp_pose=grasp_pose,
        dobot=dobot,
        gripper=gripper,
        gripper_close_pos=gripper_close_pos,
        move_speed=8,
        gripper_force=10,
        gripper_speed=30,
        verbose=True
    )

    wait_grasp = rospy.Rate(1.0 / 3)
    wait_grasp.sleep()
    
    pose_now = dobot.get_pose()
    x_adjustment = 15
    y_adjustment = 80
    z_adjustment = 70
    dobot.move_to_pose(pose_now[0]+x_adjustment, pose_now[1]+y_adjustment, pose_now[2]+z_adjustment, pose_now[3], pose_now[4], pose_now[5], speed=7, acceleration=1) 


#-----------开始检测玻璃棒方向-------------------------------------------------------
    avg_angle = detect_object_orientation(
        angle_camera=angle_camera,
        save_dir=save_dir,
        max_attempts=100,
        verbose=True
    )

    wait_detect = rospy.Rate(1.0 / 1)
    wait_detect.sleep()


#-----------开始调整玻璃棒姿态-------------------------------------------------------
    pose_target, pose_after_adjust = adjust_object_orientation(
        dobot=dobot,
        avg_angle=avg_angle,
        grasp_tilt_angle=grasp_tilt_angle,
        x_adjustment=0,
        y_adjustment=0,
        z_adjustment=-15,
        move_speed=12,
        acceleration=1,
        wait_time=9.0,
        verbose=True
    )
    

#-----------开始检测玻璃棒是否触碰到桌面-------------------------------------------------------
    print("\n开始监测玻璃棒与桌面接触...")

    gray_debug_dir = os.path.join(save_dir, "gray_images_debug")
    os.makedirs(gray_debug_dir, exist_ok=True)
    print(f"灰度图将保存到: {gray_debug_dir}")

    sample_interval = 0.1  # 秒
    move_step = 3          # mm
    max_steps = 700
    change_threshold = 3 #0.06% 变化灵敏度 

    rate = rospy.Rate(1.0 / sample_interval)
    rate.sleep()
    # rospy.sleep(sample_interval)
    frame_before = None
    while frame_before is None:
        initial_frame = contact_camera.get_current_frame()
        if initial_frame is not None:
            frame_before = initial_frame
        else:
            print("等待初始图像...")
            rospy.sleep(sample_interval)

    print("已获取初始图像")
    pose_current = dobot.get_pose()

    for step in range(max_steps):
        wait = rospy.Rate(33)  
        wait.sleep()
        # 动作前帧
        frame_data_before = contact_camera.get_current_frame()
        if frame_data_before is None:
            print(f"  步骤 {step+1}: 等待动作前图像...")
            rospy.sleep(sample_interval)
            continue
        frame_before = frame_data_before

        # 向下移动一小步
        pose_current[2] -= move_step
        dobot.move_to_pose(
            pose_current[0], pose_current[1], pose_current[2],
            pose_current[3], pose_current[4], pose_current[5],
            speed=5, acceleration=1
        )

        # 等待并抓取动作后的新帧
        frame_after = None
        has_change = False
        #连续高频采样检测
        for _ in range(20): #0.1*20 = 2s
            rate.sleep()
            candidate_frame = contact_camera.get_current_frame()
            if candidate_frame is not None:
                frame_after = candidate_frame

                has_change = contact_camera.has_significant_change(
                    frame_before, frame_after,
                    change_threshold=change_threshold,
                    pixel_threshold=2,
                    min_area=2,
                    save_dir=gray_debug_dir,
                    step_num=step
                )

                if has_change:
                    break
            
                # break

        if frame_after is None:
            print(f"  步骤 {step+1}: 未收到新图像，继续等待...")
            continue


        if has_change:
            print(f"检测到显著变化！玻璃棒可能已接触桌面 (步数: {step+1}, 下降: {(step+1)*move_step}mm)")
            break

        print(f"  步骤 {step+1}/{max_steps}: 未检测到接触，继续下降...")
    else:
        print("达到垂直向下最大移动距离，未检测到明显变化")

    print("玻璃棒下降检测完成\n")

        
    # 可选：返回home位置（根据需要取消注释）
    # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)


