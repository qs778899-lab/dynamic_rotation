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
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
from calculate_grasp_pose_from_object_pose import calculate_grasppose_from_objectpose
from calculate_grasp_pose_from_object_pose import calculate_grasppose_from_objectpose_servop



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


# ---------- 读取CSV中的center_pose（4x4矩阵，4行一组，空行分隔） ----------
def load_center_poses_from_csv(csv_path: str):
    poses = []
    current_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                # 一个pose结束
                if len(current_rows) == 4:
                    poses.append(np.array(current_rows, dtype=float))
                current_rows = []
                continue
            row = [float(x) for x in line.split(',') if x.strip() != '']
            if len(row) == 4:
                current_rows.append(row)
        # 处理文件末尾无空行的情况
        if len(current_rows) == 4:
            poses.append(np.array(current_rows, dtype=float))
    return poses


if __name__ == "__main__":
    # camera = CreateRealsense("231522072272") #已初始化相机
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
    
    # ========== 从CSV加载center_pose矩阵序列 ==========
    # csv_path = '/home/erlin/work/labgrasp/RECORD/20251023_104936/pose_array.csv'  
    csv_path = '/home/erlin/work/labgrasp/RECORD/20251023_104936/interpolation_object_pose_trajectory.csv'
    center_pose_mats = load_center_poses_from_csv(csv_path)
    print(f"center_pose_mats: ", center_pose_mats)
    
    # 间隔参数：每隔k个提取一个center_pose（k=1表示每个都用，k=2表示每隔一个提取）
    k = 1  # 轨迹提取时已经有间隔了
    
    # 当前索引
    start_pose_index = 0  #interpolation_object_pose_trajectory.csv文件中已经对起始位置处理过了？
    current_pose_index =  start_pose_index 
    # ==========================================
    
    try:
        while True:
            print("\n" + "="*60)
            # 检查是否还有未处理的pose
            if current_pose_index >= len(center_pose_mats):
                print("\n[完成] 所有center_pose已处理完毕")
                break
            
            # 获取当前的center_pose（4x4矩阵）
            center_pose = center_pose_mats[current_pose_index]
            # print(f"\n[进度] 处理第 {current_pose_index//k + 1}/{(len(center_pose_array_list)-1)//k + 1} 个pose (索引: {current_pose_index})")
            
            # key = cv2.waitKey(1)
            # if key == ord('q'):  # 按q退出
            #     break
            # elif key == ord('a'):  # 按a执行抓取
            
            center_pose_array = np.array(center_pose, dtype=float)
            print("center_pose_object_array: ", center_pose_array) 

                        
            # ------使用封装函数执行移动和抓取------
            # 配置抓取参数
            z_xoy_angle = 0  # 物体绕z轴旋转角度
            vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
            grasp_tilt_angle = 35  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度
            z_safe_distance= 55  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
            gripper_close_pos = 80 # 夹爪闭合位置 (0-1000)，值越小夹得越紧

            enable_gripper = False
            if current_pose_index == start_pose_index:
                enable_gripper = True
            
            # 调用封装函数执行抓取
            success, T_base_ee_ideal = calculate_grasppose_from_objectpose_servop(
                center_pose_array=center_pose_array,
                dobot=dobot,
                gripper=gripper,
                T_ee_cam=T_ee_cam,
                z_xoy_angle=z_xoy_angle,
                vertical_euler=vertical_euler,
                grasp_tilt_angle=grasp_tilt_angle,
                angle_threshold=10.0,
                T_tcp_ee_z=-0.17,
                T_safe_distance=0.003,
                z_safe_distance=z_safe_distance,
                gripper_close_pos=gripper_close_pos,
                enable_gripper=enable_gripper,
                verbose=True
            )
            
            
            if success:
                print("\n[成功] 本次移动完成")
            else:
                print("\n[失败] 本次移动失败")
            
            current_pose_index += k

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    # finally:
    #     cv2.destroyAllWindows()
        # dobot.disable_robot()


