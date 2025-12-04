import sys
sys.path.append("FoundationPose")


from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD   
from create_camera import CreateRealsense
import cv2
import numpy as np

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
from spatialmath import SE3, SO3

# ---------- 手眼标定 ----------
# 从相机坐标系到末端执行器坐标系的变换矩阵
T_ee_cam = SE3.Rt(
    np.array([
        [0.99994846, -0.00489269, -0.00889571],
        [0.00641473,  0.98363436,  0.18006193],
        [0.00786914, -0.18010971,  0.98361505]
    ]),
    [-0.03275156, -0.06321087, 0.07478308],
    check=False
)


if __name__ == "__main__":
    camera = CreateRealsense("317222075299")
    mesh_file = "mesh/1cm_10cm.obj"
    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_file)
    mesh.vertices /= 1000 #? 单位转换除100还是1000
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)


    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() #?
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
    depth = camera.get_frames()['depth']/1000
    ir1 = camera.get_frames()['ir1']
    ir2 = camera.get_frames()['ir2']
    cv2.imwrite("ir1.png", ir1)
    cv2.imwrite("ir2.png", ir2)
    
    #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
    mask = get_mask_from_GD(color, "red cylinder")
    cam_k = np.loadtxt(f'cam_K.txt').reshape(3,3)
    
    pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)

    print(pose)
    
    center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
    vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox) #在图像上绘制3D包围盒
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True) #在图像上绘制坐标系轴
    cv2.imshow('1', vis[...,::-1])
    cv2.waitKey(0)

    print("center_pose_object: ", center_pose) #? 这个没有被print出来？

    #-------run demo---------

