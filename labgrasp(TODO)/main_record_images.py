import sys
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD   
from create_camera import CreateRealsense
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
import json
from datetime import datetime
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose


# ---------- 手眼标定 ----------
def load_hand_eye_calibration(json_path="hand_eye_calibration.json"):
    """从JSON文件加载手眼标定矩阵"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    calibration = data['T_ee_cam']
    rotation_matrix = np.array(calibration['rotation_matrix'])
    translation_vector = calibration['translation_vector']
    return SE3.Rt(rotation_matrix, translation_vector, check=False)

# 从相机坐标系到末端执行器坐标系的变换矩阵
T_ee_cam = load_hand_eye_calibration()

# ---------- 机械臂 ----------
def init_robot():
    dobot = SimpleApi("192.168.5.1", 29999)
    dobot.clear_error()
    dobot.enable_robot()
    dobot.stop()
    # 启动力传感器
    dobot.enable_ft_sensor(1)
    time.sleep(1)
    # 力传感器置零(以当前受力状态为基准)
    dobot.six_force_home()
    time.sleep(1)
    # 力监控线程
    # force_monitor = ForceMonitor(dobot)
    # force_monitor.start_monitoring()
    # error_monitor = ErrorMonitor(dobot)
    # error_monitor.start_monitoring()
    gripper = DobotGripper(dobot)
    gripper.connect(init=True)
    return dobot, gripper


if __name__ == "__main__":
    camera = CreateRealsense("231522072272") #已初始化相机
    mesh_file = "mesh/1cm_10cm.obj"
    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_file)
    mesh.vertices /= 1000 #! 单位转换除以1000
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # 初始化机械臂
    dobot, gripper = init_robot()

    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() 
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    # 获取相机内参
    cam_k = np.loadtxt(f'cam_K.txt').reshape(3,3)
    
    # 创建保存数据的文件夹结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_base_dir = "RECORD"
    session_dir = os.path.join(record_base_dir, timestamp)
    
    # 创建子文件夹
    color_dir = os.path.join(session_dir, "color")
    depth_dir = os.path.join(session_dir, "depth")
    ir1_dir = os.path.join(session_dir, "ir1")
    ir2_dir = os.path.join(session_dir, "ir2")
    
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ir1_dir, exist_ok=True)
    os.makedirs(ir2_dir, exist_ok=True)
    
    print(f"[数据保存] 保存路径: {session_dir}")
    
    try:
        frame_count = 0
        
        while True:
            print(f"[数据保存] 开始保存第 {frame_count} 帧")
            # 获取当前帧
            color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            depth = camera.get_frames()['depth']/1000
            ir1 = camera.get_frames()['ir1']
            ir2 = camera.get_frames()['ir2']
            
            # 保存当前帧数据到对应的子文件夹
            frame_filename = f"frame_{frame_count:06d}.png"
            cv2.imwrite(os.path.join(color_dir, frame_filename), color)
            cv2.imwrite(os.path.join(ir1_dir, frame_filename), ir1)
            cv2.imwrite(os.path.join(ir2_dir, frame_filename), ir2)
            
            # 保存深度图（需要转换回uint16格式以保留精度）
            depth_uint16 = (depth * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, frame_filename), depth_uint16)
            
            frame_count += 1
            
            if frame_count % 30 == 0:  # 每30帧打印一次
                print(f"[数据保存] 已保存 {frame_count} 帧")
                
            # 可选：返回home位置（根据需要取消注释）
            # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)
    

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        print(f"\n[数据保存] 总共保存了 {frame_count} 帧到 {session_dir}")
        cv2.destroyAllWindows()
        # dobot.disable_robot()

    #-------run demo---------

