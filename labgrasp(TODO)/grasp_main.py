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
    
    
    try:
        frame_count = 0
        last_valid_pose = None  # 保存上一次有效的pose
        
        while True:
            # 获取当前帧
            color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            depth = camera.get_frames()['depth']/1000
            ir1 = camera.get_frames()['ir1']
            ir2 = camera.get_frames()['ir2']
            cv2.imwrite("ir1.png", ir1)
            cv2.imwrite("ir2.png", ir2)
            
            # 每隔70帧进行一次FoundationPose检测
            if frame_count % 70 == 0:
                #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
                mask = get_mask_from_GD(color, "red cylinder")
                pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
                print(f"第{frame_count}帧检测完成，pose: {pose}")
                center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
                vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('1', vis[...,::-1])
                # cv2.waitKey(0) #waitKey(0) 是一种阻塞
                # input("break001") #input也是一种阻塞
                # print("break001")
                last_valid_pose = center_pose  # 保存这次检测的结果
            else:
                # 使用上一次检测的结果
                center_pose = last_valid_pose
                print(f"第{frame_count}帧使用上次检测结果")
            

            print("center_pose_object: ", center_pose) 
            frame_count += 1

            key = cv2.waitKey(1)
            # if key == ord('q'):  # 按q退出
            #     break
            # elif key == ord('a'):  # 按a执行抓取
            
            # 将center_pose转换为numpy数组
            center_pose_array = np.array(center_pose, dtype=float)
            
            # ------使用封装函数执行抓取------
            # 配置抓取参数
            z_xoy_angle = 30  # 物体绕z轴旋转角度
            vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
            grasp_tilt_angle = 55  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度
            grasp_tilt_angle = 40  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度
            z_safe_distance= 30  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
            
            # 调用封装函数执行抓取
            success, T_base_ee_ideal = execute_grasp_from_object_pose(
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
                gripper_close_pos=80,
                verbose=True
            )
            
            if success:
                print("\n[成功] 抓取操作完成!")
                input("按Enter继续...")
            else:
                print("\n[失败] 抓取操作未完成")
                
            # 可选：返回home位置（根据需要取消注释）
            # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)
    

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        cv2.destroyAllWindows()
        # dobot.disable_robot()

    #-------run demo---------

