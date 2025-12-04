# coding: utf-8
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue

# ---------- GraspNet ----------
# sys.path.append("/home/erlin/YOLO_World-SAM-GraspNet/graspnet-baseline/models")
# sys.path.append("/home/erlin/YOLO_World-SAM-GraspNet/graspnet-baseline/dataset")
# sys.path.append("/home/erlin/YOLO_World-SAM-GraspNet/graspnet-baseline/utils")
'''
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector
'''
from spatialmath import SE3, SO3

# ---------- 常量 ----------
CHECKPOINT = "/home/erlin/TCP-IP-Python-V4/logs/log_rs/checkpoint-rs.tar"
SAM_WEIGHT = "/home/erlin/TCP-IP-Python-V4/sam_b.pt"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 手眼标定 ----------
T_ee_cam = SE3.Rt(
    np.array([
        [0.99994846, -0.00489269, -0.00889571],
        [0.00641473,  0.98363436,  0.18006193],
        [0.00786914, -0.18010971,  0.98361505]
    ]),
    [-0.03275156, -0.06321087, 0.07478308],
    check=False
)

# ---------- 相机 ----------
def init_camera():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.depth)
     
    # 自动曝光
    for _ in range(30): pipeline.wait_for_frames()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    return pipeline, align, profile, depth_scale

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
    force_monitor = ForceMonitor(dobot)
    force_monitor.start_monitoring()
    error_monitor = ErrorMonitor(dobot)
    error_monitor.start_monitoring()
    gripper = DobotGripper(dobot)
    gripper.connect(init=True)
    return dobot, gripper



def main():
    print("初始化中...")

    dobot, gripper = init_robot()

    try:
        while True:
            
            # frames = pipeline.wait_for_frames()
            # aligned = align.process(frames)

            # color_frame = aligned.get_color_frame()
            # depth_frame = aligned.get_depth_frame()

            # color = np.asanyarray(color_frame.get_data())
            # depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            # color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # cv2.imshow('RGB', color_bgr)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break
            best_grasp = None

            # while retry_count < max_retries:
            #     retry_count += 1
            #     print(f"\n[DEBUG] 第 {retry_count} 次尝试获取抓取姿态...")

            '''
            mask = click_to_mask(color_bgr, x_click, y_click)       #点击目标物体(简化实验条件？)
            pts = mask_to_pts(depth, mask, color_intr, depth_scale) #生成目标物体点云
            if len(pts) < 100:
                print("[WARNING] 点云数量不足100，跳过")
                continue

            pts_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pts_o3d, _ = pts_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pts_clean = np.asarray(pts_o3d.points)dedpde #无序点云集合
            '''

            '''
            gg = grasp_from_pts(net, pts_clean) #返回的是抓取姿态
            if gg is None or len(gg) == 0:
                print("[WARNING] 无可用抓取姿态，重试中...")
                continue
            best_grasp = select_best_grasp(gg, pts_clean)
            '''





            # best_grasp = [[1, 0, 0, 0.1],
            #             [0, -1, 0, 0.2],
            #             [0, 0, -1, 0.3],
            #             [0, 0, 0, 1]]

            best_grasp = [[0.8155072, 0.57407063, -0.07343091, 0.1],
                                [0.57838976, -0.8128748, 0.06854292, 0.2],
                                [-0.02034182, -0.09836861, -0.99494296, 0.3+0.1],
                                [0, 0, 0, 1]]
            
                
                # if best_grasp is not None:
                #     vis_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
                #     vis_cloud.colors = o3d.utility.Vector3dVector(
                #         np.tile([0.5, 0.5, 0.5], (pts_clean.shape[0], 1)))
                #     arrow = best_grasp.to_open3d_geometry(color=[1, 0, 0])
                #     o3d.visualization.draw_geometries([vis_cloud, arrow], window_name="Grasp Preview")
                #     break
            print("best grasp",best_grasp)  #best grasp [[0.8155072, 0.57407063, -0.07343091, 0.1], [0.57838976, -0.8128748, 0.06854292, 0.2], [-0.02034182, -0.09836861, -0.99494296, 0.8], [0, 0, 0, 1]]


            # 转换为相机坐标系下的抓取姿态
            T_cam_grasp = SE3(best_grasp, check=False) * \
                          SE3.Ry(90, unit='deg') * \
                          SE3.Rz(-90, unit='deg')
            # 末端执行器坐标系到抓取点的变换： 抓取相对于末端 + 夹爪Z偏移0.20m
            T_ee_grasp = T_ee_cam * T_cam_grasp * SE3(0, 0, -0.17)

            # 当前机器人基坐标系位姿（保持与导出RPY顺序一致）
            pose_now = dobot.get_pose()
            print("posenow",pose_now)  #posenow [611.6507, -184.4319, 405.1806, -169.8548, 0.7272, -98.6116]
            x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
            # 基座坐标系到末端执行器坐标系的变换
            T_base_ee = SE3.Rt(
                SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
                np.array([x_e, y_e, z_e]) / 1000.0,
                check=False
            )

            # 基座坐标系到抓取点的变换
            T_base_grasp = T_base_ee * T_ee_grasp   

            pos_mm = T_base_grasp.t * 1000   #提取变换矩阵的平移部分      
            rx, ry, rz = T_base_grasp.rpy(unit='deg', order='zyx')

            if abs(rz) > 100: #角度限制，等价转换
                rz = -(180 - rz) if rz > 0 else 180 + rz
            dobot.move_to_pose(*pos_mm, rx, ry, rz) 
            if dobot.check_pose(*pos_mm): #control(position, force, speed)
                gripper.control(1000, 30, 30) #执行抓取动作
            input("break")
            time.sleep(3)
            dobot.move_to_pose(pos_mm[0], pos_mm[1], pos_mm[2]-98, rx, ry, rz)

            input("break")
            if dobot.check_pose(pos_mm[0], pos_mm[1], pos_mm[2]-98):
                gripper.control(100, 30, 30)
            dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524,speed=30,acceleration=30)
            

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        # pipeline.stop()
        cv2.destroyAllWindows()
        dobot.disable_robot()

if __name__ == "__main__":
    main()

