#!/usr/bin/env python3
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
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



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


#??? ros 配置可以封装进另一个函数里吗？
# ---------- ROS节点 ----------
class ROSSubscriberTest:
    def __init__(self):
        """初始化ROS节点和订阅者"""
        rospy.init_node('ros_subscriber_test', anonymous=True)  ##ros node name 只是告诉 ROS：“我这个节点叫什么”，与任何话题名或函数名没有直接绑定关系；保持唯一性即可。
        
        # 初始化cv_bridge用于图像转换
        self.bridge = CvBridge()

        #mark: 整体的ros接收信息方案： callback_function + 缓存(atest_tracking_data,latest_image) + 访问函数(get_latest_tracking_data,get_latest_image)
        
        # 缓存最新的tracking_data
        self.latest_tracking_data = {
            'angle_z_deg': 0.0,
            'b': 0.0,
            'x': 0.0,
            'y': 0.0,
            'timestamp': 0.0,
            'valid': False
        }
        self.data_lock = threading.Lock()
        
        # 缓存最新的image
        self.latest_image = None
        self.image_timestamp = 0.0
        self.image_lock = threading.Lock()
        
        # 缓存最新的原始图像（用于图像处理）
        self.latest_raw_image = None
        self.raw_image_timestamp = 0.0
        self.raw_image_lock = threading.Lock()

        
        # 订阅tracking_data topic
        self.tracking_sub = rospy.Subscriber(
            'tracking_data', #topic name: tracking_data
            Float64MultiArray, 
            self.tracking_callback #mark: callback_function 
        )
        
        # 订阅object_orientation topic  
        self.image_sub = rospy.Subscriber(
            'image_object_orientation', #topic name: image_object_orientation
            Image,
            self.image_callback
        )
        
        # 订阅raw_image topic (纯净原始图像，用于图像处理)
        self.raw_image_sub = rospy.Subscriber(
            'raw_image', #topic name: raw_image
            Image,
            self.raw_image_callback
        )
        
        print("ROS订阅者已启动，等待数据...")
        
    def tracking_callback(self, msg):
        """处理tracking_data消息"""
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            b = msg.data[1] 
            x = msg.data[2]
            y = msg.data[3]

            with self.data_lock:
                self.latest_tracking_data.update({
                    'angle_z_deg': angle_z_deg,
                    'b': b,
                    'x': x,
                    'y': y,
                    'timestamp': time.time(),
                    'valid': True
                })
            
            # print(f"📊 跟踪数据: 角度={angle_z_deg:.2f}°, 截距={b:.6f}, 位置=({x:.6f}, {y:.6f})")
        else:
            pass
            # print(f"⚠️  跟踪数据格式错误，期望4个值，实际收到{len(msg.data)}个值")
    
    def image_callback(self, msg):
        """处理object_orientation图像消息"""
        try:
            # print(f"[DEBUG] 图像回调被触发！消息类型: {type(msg)}")
            # print(f"[DEBUG] 图像编码: {msg.encoding}, 尺寸: {msg.width}x{msg.height}")
            
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 线程安全地保存图像
            with self.image_lock:
                self.latest_image = cv_image.copy()
                self.image_timestamp = time.time()
            
            # # 显示图像
            # cv2.imshow("Object Orientation", cv_image)
            # cv2.waitKey(1)  # 非阻塞等待，允许其他处理
            height, width = cv_image.shape[:2]
            # print(f"🖼️  成功接收并保存图像: {width}x{height} pixels")
            
        except Exception as e:
            print(f"❌ 图像处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def raw_image_callback(self, msg):
        """处理raw_image原始图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 线程安全地保存原始图像
            with self.raw_image_lock:
                self.latest_raw_image = cv_image.copy()
                self.raw_image_timestamp = time.time()
            
            height, width = cv_image.shape[:2]
            # print(f"📷 成功接收原始图像: {width}x{height} pixels")
            
        except Exception as e:
            # Fallback: 手动解析ROS Image消息（绕过cv_bridge的libffi问题）
            print(f"❌ raw 图像处理错误: {e}")
            
    
    #mark: 在callback_function基础上，访问缓存的最新数据
    def get_latest_tracking_data(self):
        """获取最新的跟踪数据（线程安全）"""
        with self.data_lock:
            return self.latest_tracking_data.copy()
    
    def get_latest_image(self):
        """获取最新的图像数据（线程安全）"""
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy(), self.image_timestamp
            else:
                return None, 0.0
    
    def get_latest_raw_image(self):
        """获取最新的原始图像数据（线程安全）"""
        with self.raw_image_lock:
            if self.latest_raw_image is not None:
                return self.latest_raw_image.copy(), self.raw_image_timestamp
            else:
                return None, 0.0
    
    def run(self):
        """运行订阅者"""
        try:
            # 非阻塞保活循环：等待ROS事件，但不主动退出
            while not rospy.is_shutdown():
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n ros中断,正在退出...")


if __name__ == "__main__":
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"图像将保存到: {save_dir}")
    
    # 创建角度数据记录文件
    angle_log_path = os.path.join(save_dir, "angle_log.csv")
    with open(angle_log_path, 'w') as f:
        f.write("frame,timestamp,angle_z_deg,detected_angles,avg_angle\n")
    # print(f"角度数据将保存到: {angle_log_path}")
    
    camera = CreateRealsense("231522072272") 
    # mesh_file = "mesh/cube.obj"
    mesh_file = "mesh/thin_cube.obj"
    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_file)
    #? openscad的单位是mm， 但是转为obj文件后单位又变成m，所以还是需要转换！
    mesh.vertices /= 1000 #! 单位转换除以1000
    # mesh.vertices /= 3
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # 初始化机械臂
    dobot, gripper = init_robot()

    #? 初始化ROS订阅者（在后台daemon线程运行，不会阻塞main程序）
    try:
        ros_subscriber = ROSSubscriberTest()
        ros_thread = threading.Thread(target=ros_subscriber.run, daemon=True)
        ros_thread.start()
        print("✅ ROS订阅者已在后台启动（非阻塞模式）")
        time.sleep(1)  # 短暂等待ROS节点启动
    except Exception as e:
        print(f"⚠️  ROS订阅者启动失败: {e}")
        # 创建一个空的占位对象，防止后续代码出错
        class DummySubscriber:
            def get_latest_tracking_data(self):
                return {'valid': False, 'angle_z_deg': 0.0, 'b': 0.0, 'x': 0.0, 'y': 0.0, 'timestamp': 0.0}
        ros_subscriber = DummySubscriber()

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
        last_valid_angle = None  # 保存上一次有效的ROS角度
        last_seen_ts = None  # 上一次使用的ROS时间戳（tracking_data）
        last_seen_img_ts = None  # 上一次使用的图像时间戳
        last_valid_detected_angles = None  # 保存上一次检测到的角度列表
        last_valid_avg_angle = 0.0  # 保存上一次检测到的平均角度
        
        while True:
            # 获取当前帧
            # color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            # depth = camera.get_frames()['depth']/1000
            # ir1 = camera.get_frames()['ir1']
            # ir2 = camera.get_frames()['ir2']
            frames = camera.get_frames()
            if frames is None:
                continue
            color = frames['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            depth = frames['depth']/1000
            ir1 = frames['ir1']
            ir2 = frames['ir2']

            color_path = os.path.join(save_dir, f"color_frame_{frame_count:06d}.png")
            print("befor foundation pose, color_shape: ", color.shape)
            cv2.imwrite(color_path, color)
            
            
            # 每隔30帧进行一次FoundationPose检测
            if frame_count % 15 == 0:
                #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
                mask = get_mask_from_GD(color, "red stirring rod")
                # mask = get_mask_from_GD(color, "Plastic dropper") 
                # mask = get_mask_from_GD(color, "long yellow bar")
                # mask = get_mask_from_GD(color, "long red bar")
                # print("mask_shape: ", mask.shape)
            
                cv2.imshow("mask", mask)
                cv2.imshow("color", color)
                pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
                print(f"第{frame_count}帧检测完成，pose: {pose}")
                center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
                vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('1', vis[...,::-1])
    
                mask_path = os.path.join(save_dir, f"mask_frame_{frame_count:06d}.png")
                vis_path = os.path.join(save_dir, f"vis_frame_{frame_count:06d}.png")
                cv2.imwrite(mask_path, mask)
                cv2.imwrite(vis_path, vis[...,::-1])                

                # cv2.waitKey(0) #waitKey(0) 是一种阻塞
                # input("break001") #input也是一种阻塞
                # print("break001")
                
                #? 清理内存 (这个有用吗？)
                torch.cuda.empty_cache()
                gc.collect()
 
                last_valid_pose = center_pose  # 保存这次检测的结果
            else:
                # 使用上一次检测的结果
                center_pose = last_valid_pose
                # print(f"第{frame_count}帧使用上次检测结果")
            

            print("center_pose_object: ", center_pose) 
            
            frame_count += 1

            if center_pose is not None:
                break

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        cv2.destroyAllWindows()
        # dobot.disable_robot()


    key = cv2.waitKey(1)
    # if key == ord('q'):  # 按q退出
    #     break
    # elif key == ord('a'):  # 按a执行抓取
    #
    # init_position = 10
    # gripper.control(position=init_position, force=80, speed=10)


    #mark: 获取ROS跟踪数据（非阻塞）
    tracking_data = ros_subscriber.get_latest_tracking_data()
    has_new_msg = tracking_data['valid'] and (
        last_seen_ts is None or tracking_data['timestamp'] > last_seen_ts
    )
    if has_new_msg:
        # 收到新ROS数据，更新并使用最新角度
        angle_z_deg = tracking_data['angle_z_deg']
        last_valid_angle = angle_z_deg
        last_seen_ts = tracking_data['timestamp']
        print(f"🔄 使用ROS跟踪角度: {angle_z_deg:.2f}° ")
    else:
        # 没有新ROS数据
        if last_valid_angle is not None:
            angle_z_deg = last_valid_angle
            print(f"使用上次ROS角度: {angle_z_deg:.2f}° (当前无新数据)")
        else:
            angle_z_deg = -45  # 朝里
            print("从未接收到ROS数据，使用默认角度: -45°")


    # 将center_pose转换为numpy数组
    center_pose_array = np.array(center_pose, dtype=float)
    
    # ------使用封装函数执行抓取------
    # 配置抓取参数
    z_xoy_angle = 0 # 物体绕z轴旋转角度
    vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
    grasp_tilt_angle = 30  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度： 加了30度会朝外旋转
    z_safe_distance= 39  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
    
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
        T_tcp_ee_z= -0.16, 
        T_safe_distance= 0.00, #可灵活调整
        z_safe_distance=z_safe_distance,
        gripper_close_pos=20,
        verbose=True
    )
    
    pose_now = dobot.get_pose()
    x_adjustment = 115
    z_adjustment = 180
    dobot.move_to_pose(pose_now[0]+x_adjustment, pose_now[1], pose_now[2]+z_adjustment, pose_now[3], pose_now[4], pose_now[5], speed=7, acceleration=1) 


    #mark: 循环获取ROS原始图像并检测方向，直到检测成功
    print("\n" + "="*60)
    print("🔍 开始检测玻璃棒方向...")
    print("="*60)
    
    detected_angles = None
    avg_angle = 0.0
    detection_attempts = 0
    
    while True:
        detection_attempts += 1
        
        # 获取ROS原始图像数据
        raw_image, img_timestamp = ros_subscriber.get_latest_raw_image()
        has_new_image = raw_image is not None
        
        if has_new_image:
            # 收到新图像，进行方向检测
            print(f"\n📷 第{detection_attempts}次尝试: 检测新原始图像方向 (时间戳: {img_timestamp:.2f})")
            detected_angles, avg_angle = detect_dent_orientation(raw_image, save_dir=save_dir)
            
            if detected_angles:
                last_valid_detected_angles = detected_angles
                last_valid_avg_angle = avg_angle
                last_seen_img_ts = img_timestamp
                print(f"成功检测到物体朝向角度: {detected_angles}, 平均: {avg_angle:.2f}°")
                print("="*60)
                break  
            else:
                print("当前图像未检测到明显方向特征，继续等待...")
                time.sleep(0.1)  
        else:
            print(f"第{detection_attempts}次尝试: 等待图像数据...")
            time.sleep(0.1)  
        
        # 可选：最大尝试次数限制
        if detection_attempts >= 100:
            print(" 警告: 达到最大尝试次数(100次)，使用默认角度")
            detected_angles = []
            avg_angle = 0.0
            break

    # 记录angle_z_deg 和 detected_angles到log文件
    with open(angle_log_path, 'a') as f:
        angles_str = str(detected_angles) if detected_angles is not None else "None"
        f.write(f"{frame_count},{time.time():.3f},{angle_z_deg:.2f},{angles_str},{avg_angle:.2f}\n")



#-----------开始调整玻璃棒姿态-------------------------------------------------------

    print("开始调整玻璃棒姿态至垂直桌面向下")
    pose_now = dobot.get_pose()
    delta_ee = avg_angle - grasp_tilt_angle
    #需要让tcp朝外旋转； grasp_tilt_angle为正值时，tcp会朝外旋转。
    pose_target = [pose_now[0]+15, pose_now[1], pose_now[2], pose_now[3]+delta_ee, pose_now[4], pose_now[5]]
    dobot.move_to_pose(pose_target[0], pose_target[1], pose_target[2], pose_target[3], pose_target[4], pose_target[5], speed=12, acceleration=1)
    

    wait_rate = rospy.Rate(1.0 / 12.0)  
    wait_rate.sleep()
    
    # 验证是否到达目标位置
    pose_after_adjust = dobot.get_pose()
    print(f"检查姿态调整是否完成: Rx={pose_after_adjust[3]:.2f}° (目标: {pose_target[3]:.2f}°)")

    #垂直桌面向下移动玻璃棒，检测是否触碰到桌面
    print("\n开始监测玻璃棒与桌面接触...")

    move_step = 1          # mm
    max_steps = 700
    sample_interval = 0.03  # 秒
    max_force_samples = 30
    force_threshold = 1.0  # N，触碰判定阈值
    consecutive_hits_required = 2

    pose_current = dobot.get_pose()
    contact_detected = False
    contact_force = 0.0

    for step in range(max_steps):
        wait = rospy.Rate(33)
        wait.sleep()

        pose_current[2] -= move_step
        dobot.move_to_pose(
            pose_current[0], pose_current[1], pose_current[2],
            pose_current[3], pose_current[4], pose_current[5],
            speed=5, acceleration=1
        )

        consecutive_hits = 0
        for _ in range(max_force_samples):
            short_wait = rospy.Rate(1/sample_interval)
            short_wait.sleep()
            force_values = dobot.get_force()
            if not force_values:
                continue

            print("force_values: ", force_values)
            

            max_force_component = max(abs(value) for value in force_values)
            if max_force_component >= force_threshold:
                consecutive_hits += 1
                contact_force = max_force_component
                if consecutive_hits >= consecutive_hits_required:
                    contact_detected = True
                    break
            else:
                consecutive_hits = 0

        if contact_detected:
            print(
                f"检测到受力变化！玻璃棒可能已接触桌面 (步数: {step+1}, 下降: {(step+1)*move_step}mm, Fz≈{contact_force:.2f}N)"
            )
            break

        print(f"  步骤 {step+1}/{max_steps}: 未检测到接触，继续下降...")
    else:
        print("达到垂直向下最大移动距离，未检测到明显受力变化")

    print("玻璃棒下降检测完成\n")

        
    # 可选：返回home位置（根据需要取消注释）
    # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)

    #移动到目标位置
    pose_now = dobot.get_pose()
    x_target, y_target, z_target= 450, -150, 12
    rx_target, ry_target, rz_target= pose_now[3], pose_now[4], pose_now[5]
    # dobot.move_to_pose(x_target, y_target, z_target, rx_target, ry_target, rz_target, speed=9)


