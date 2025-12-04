import argparse
import os

import cv2
import numpy as np
import yaml
from sklearn.decomposition import PCA

# ROS imports
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow, InsufficientOverlapError
from normalflow.utils import erode_contact_mask, gxy2normal, transform2pose
from normalflow.viz_utils import annotate_coordinate_system

"""
This script demonstrates real-time object tracking using normalflow with automatic reference frame generation.
The system automatically detects the principal axis of the contact object and sets up a coordinate system
with x-axis along the principal axis (kept in the right half of the image) and y-axis perpendicular to it.
"""

# --- Configuration Constants ---
DEFAULT_CALIB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")

MIN_CONTACT_AREA_PIXELS = 500
BACKGROUND_IMAGE_COLLECTION_COUNT = 10

# Error thresholds for resetting the keyframe (values are in degrees for rotation, mm for translation)
RESET_ROT_THRESHOLD = 5.0
RESET_TRANS_THRESHOLD = 1.0

def init_ros_publisher():
    """Initialize ROS node and publishers for tracking data and object orientation image."""
    rospy.init_node('object_tracking_publisher', anonymous=True)
    #!一个ros node发送三个ros topic
    pub_tracking = rospy.Publisher('tracking_data', Float64MultiArray, queue_size=10)
    pub_image = rospy.Publisher('image_object_orientation', Image, queue_size=10)
    pub_raw_image = rospy.Publisher('raw_image', Image, queue_size=10)
    bridge = CvBridge()
    return pub_tracking, pub_image, pub_raw_image, bridge

def publish_tracking_data(publisher, angle_z_deg, b, x, y):
    """Publish angle_z_deg and b values via ROS."""
    if publisher is not None:
        msg = Float64MultiArray()
        msg.data = [angle_z_deg, b, x, y]
        publisher.publish(msg)
        
        # 醒目的ROS数据发送打印
        print("=" * 60)
        print("🚀 ROS TOPIC PUBLISHED: tracking_data")
        print("=" * 60)
        print(f"📊 角度 (angle_z_deg): {angle_z_deg:.6f}°")
        print(f"📏 截距 (b):           {b:.6f}")
        print(f"📍 X坐标 (x):          {x:.6f}")
        print(f"📍 Y坐标 (y):          {y:.6f}")
        print("=" * 60)
        
        rospy.loginfo(f"Published tracking data - angle_z_deg: {angle_z_deg:.6f}, b: {b:.6f}, x: {x:.6f}, y: {y:.6f}")

def publish_image(publisher, bridge, image):
    """Publish image via ROS."""
    if publisher is not None and image is not None:
        try:
            img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            publisher.publish(img_msg)
            
            # 醒目的ROS图像发送打印
            height, width = image.shape[:2]
            print("=" * 60)
            print("🖼️  ROS TOPIC PUBLISHED: image object_orientation")
            print("=" * 60)
            print(f"📐 图像尺寸: {width} x {height} pixels")
            print(f"🎨 编码格式: bgr8")
            print(f"💾 数据大小: {len(img_msg.data)} bytes")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ ROS图像发布失败: {e}")
            rospy.logerr(f"Failed to publish image: {e}")

def publish_raw_image(publisher, bridge, image):
    """Publish raw image via ROS (without any annotations)."""
    if publisher is not None and image is not None:
        try:
            # 尝试使用cv_bridge转换
            img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            publisher.publish(img_msg)
            
            # 醒目的ROS图像发送打印
            height, width = image.shape[:2]
            print("=" * 60)
            print("📷 ROS TOPIC PUBLISHED: raw_image")
            print("=" * 60)
            print(f"📐 图像尺寸: {width} x {height} pixels")
            print(f"🎨 编码格式: bgr8")
            print(f"💾 数据大小: {len(img_msg.data)} bytes")
            print("=" * 60)
            
        except Exception as e:
            print(f"ros raw image 发送失败")
        
class ReferenceFrame:
    """Container for reference frame data"""
    def __init__(self, image, H, C, N, cx, cy, principal_angle=0.0):
        self.image = image
        self.H = H
        self.C = C
        self.N = N
        self.cx = cx
        self.cy = cy
        self.principal_angle = principal_angle  
        
    def get_coordinate_system_vectors(self):
        """Get coordinate system unit vectors for visualization"""
        cos_a = np.cos(self.principal_angle)
        sin_a = np.sin(self.principal_angle)

        x_vec = np.array([cos_a, sin_a])
        y_vec = np.array([-sin_a, cos_a])  
        
        return np.array([x_vec, y_vec, [0, 0]])
        
    def get_coordinate_transform_matrix(self):
        """Get coordinate transformation matrix"""
        T = np.eye(4)
        cos_a = np.cos(self.principal_angle)
        sin_a = np.sin(self.principal_angle)
        T[0, 0] = cos_a
        T[0, 1] = -sin_a
        T[1, 0] = sin_a
        T[1, 1] = cos_a
        return T

def generate_reference_from_first_frame(image_curr, reconstructor, ppmm, min_contact_area=500):
    """
    从第一帧图像生成单个参考图像，自动检测长条物体的长边方向作为初始x方向。
    
    Args:
        image_curr: 当前帧图像
        reconstructor: 重建器对象
        ppmm: 像素到毫米的转换比例
        min_contact_area: 最小接触面积阈值
        
    Returns:
        ReferenceFrame: 单个参考帧对象，或者 None: 如果检测失败
    """
    
    # 获取表面信息
    G_curr, H_curr, C_curr = reconstructor.get_surface_info(image_curr, ppmm)
    C_curr = erode_contact_mask(C_curr)
    
    # 检查接触面积
    if np.sum(C_curr) < min_contact_area:
        print(f"接触面积太小 ({np.sum(C_curr)} < {min_contact_area} pixels)")
        return None
    
    # 计算法向量
    N_curr = gxy2normal(G_curr)
    
    # 计算接触区域的质心
    cx_curr, cy_curr = calculate_contact_centroid(C_curr)
    if cx_curr is None or cy_curr is None:
        print("无法计算接触区域质心")
        return None
    
    # 检测长条物体的主轴方向
    principal_angle = detect_principal_axis(C_curr)
    if principal_angle is None:
        print("无法检测到主轴方向")
        return None
    
    # 调整角度，确保x轴在1,4象限（右半平面）
    adjusted_angle = adjust_angle_to_right_half(principal_angle)
    
    print(f"检测到的主轴角度: {np.degrees(principal_angle):.2f}°")
    print(f"调整后的角度: {np.degrees(adjusted_angle):.2f}°")
    
    # 生成单个参考帧，长边方向为x轴
    reference = ReferenceFrame(
        image=image_curr.copy(),
        H=H_curr.copy(),
        C=C_curr.copy(),
        N=N_curr.copy(),
        cx=cx_curr,
        cy=cy_curr,
        principal_angle=adjusted_angle
    )
    
    return reference

def detect_principal_axis(contact_mask):
    """
    检测接触区域的主轴方向（长边方向）。
    
    Args:
        contact_mask: 接触掩码（二值图像）
        
    Returns:
        float: 主轴角度（弧度），如果检测失败则返回None
    """
    
    # 找到接触区域的轮廓
    binary_mask = (contact_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 获取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 方法1：使用轮廓的最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    angle_deg = rect[2]  # 角度（度）
    width, height = rect[1]
    
    # 确保角度对应长边方向
    if width < height:
        angle_deg += 90
    
    # 转换为弧度
    angle_rad = np.radians(angle_deg)
    
    # 方法2：使用PCA进行验证
    # 获取轮廓点
    points = largest_contour.reshape(-1, 2)
    
    # 使用PCA找到主成分
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # 第一主成分对应最大方差方向（通常是长边）
    principal_component = pca.components_[0]
    pca_angle = np.arctan2(principal_component[1], principal_component[0]) #np.arctan2(y,x)
    
    # 比较两种方法的结果
    angle_diff = abs(angle_rad - pca_angle)
    if angle_diff > np.pi/2:
        angle_diff = np.pi - angle_diff
    
    # # 如果两种方法差异较大，优先使用PCA结果
    # if angle_diff > np.pi/4:
    #     print(f"轮廓方法角度: {np.degrees(angle_rad):.2f}°, PCA方法角度: {np.degrees(pca_angle):.2f}°")
    #     print("使用PCA方法结果")
    #     return pca_angle
    # else:
    #     return angle_rad

    return pca_angle

def adjust_angle_to_right_half(angle_rad):
    """
    调整角度，确保x轴方向在图像右半平面。
    将角度映射到[-π/2, π/2]范围，保持x轴指向右侧。
    """
    # 将角度标准化到[-π, π]
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    
    # 如果角度在左半平面（|angle| > π/2），调整到右半平面
    if angle_rad > np.pi / 2:
        angle_rad = angle_rad - np.pi
    elif angle_rad < -np.pi / 2:
        angle_rad = angle_rad + np.pi
        
    return angle_rad

def get_coordinate_vectors_from_angle(angle_rad, scale=1.0):
    """
    根据给定角度获取坐标系向量。
    返回用于可视化的x和y单位向量。
    """
    # x轴方向
    x_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * scale
    # y轴方向（垂直于x轴）
    y_vec = np.array([-np.sin(angle_rad), np.cos(angle_rad)]) * scale
    
    return np.array([x_vec, y_vec, [0, 0]])

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time tracking the object using tactile sensors with automatic reference detection."
    )
    parser.add_argument(
        "-b",
        "--calib_model_path",
        type=str,
        help=f"Directory where calibration data and model are stored (default: {DEFAULT_CALIB_MODEL_PATH})",
        default=DEFAULT_CALIB_MODEL_PATH,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help=f"Path of the configuration file for the GelSight sensor (default: {DEFAULT_CONFIG_PATH})",
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "-s",
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        help="The sensor streamer. 'opencv' for 'gs_device.Camera' and 'ffmpeg' for 'gs_device.FastCamera'.",
        default="opencv",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="The device to run the neural network model that predicts the normal map (default: 'cpu')",
        default="cpu",
    )
    return parser.parse_args()

def load_sensor_config(config_path):
    """Loads sensor configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        required_keys = ["device_name", "ppmm", "imgh", "imgw", "raw_imgh", "raw_imgw", "framerate"]
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing one or more required keys in config file: {required_keys}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        exit()
    except ValueError as e:
        print(f"Error in configuration file: {e}")
        exit()

def connect_to_sensor(streamer_type, config):
    """Connects to the GelSight sensor based on streamer type."""
    device_name = config["device_name"]
    imgh, imgw = config["imgh"], config["imgw"]
    raw_imgh, raw_imgw = config["raw_imgh"], config["raw_imgw"]
    framerate = config["framerate"]

    if streamer_type == "opencv":
        device = Camera(device_name, imgh, imgw)
    elif streamer_type == "ffmpeg":
        device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    else:
        raise ValueError(f"Unknown streamer type: {streamer_type}")

    try:
        device.connect()
        print(f"Successfully connected to GelSight sensor: {device_name}")
        return device
    except Exception as e:
        print(f"Error connecting to GelSight sensor: {e}")
        exit()

def collect_background_images(device, reconstructor, count=BACKGROUND_IMAGE_COLLECTION_COUNT):
    """Collects and loads background images for reconstruction."""
    print(f"Collecting {count} background images, please wait ...")
    bg_images = []
    for i in range(count):
        image = device.get_image()
        if image is None:
            print(f"Warning: Could not get image from device during background collection (attempt {i+1}/{count}).")
            continue
        bg_images.append(image)
    
    if not bg_images:
        print("Error: Failed to collect any background images. Cannot proceed with reconstruction.")
        device.release()
        exit()

    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
    reconstructor.load_bg(bg_image)
    print("Done with background collection.")

def calculate_contact_centroid(contact_mask, debug_image=None):
    """Calculate the centroid of the contact area from the contact mask."""
    binary_mask = (contact_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if debug_image is not None:
        cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawContours(debug_image, [largest_contour], -1, (255, 255, 0), 2)
        cv2.imshow("Debug Centroid", debug_image)
    
    return cx, cy

def realtime_object_tracking():
    args = parse_arguments()

    # Initialize ROS publishers
    ros_pub_tracking, ros_pub_image, ros_pub_raw_image, ros_bridge = init_ros_publisher()

    # Load configuration
    config = load_sensor_config(args.config_path)
    ppmm = config["ppmm"] # This is mm/pixel as clarified by user
    imgh, imgw = config["imgh"], config["imgw"]

    # Connect to the sensor and the reconstructor
    device = connect_to_sensor(args.streamer, config)
    recon = Reconstructor(args.calib_model_path, device=args.device)

    # Collect background images
    collect_background_images(device, recon)

    # 等待第一次接触并生成参考图像
    print("等待接触，将从第一帧自动生成参考图像...")
    reference = None
    is_reset = False
    while True:
        first_frame = device.get_image()
        if first_frame is None:
            continue
        
        reference = generate_reference_from_first_frame(first_frame, recon, ppmm, MIN_CONTACT_AREA_PIXELS)
        if reference is not None:
            print("成功生成参考图像！")
            break
        else:
            print("等待有效接触...")
            # 只显示左侧图像，显示"waiting for contact"
            display_image = np.zeros((imgh, imgw * 2, 3), dtype=np.uint8)
            cv2.putText(
                display_image,
                "Waiting for Contact",
                (imgw // 2, imgh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.imshow("frame", display_image)
            if cv2.waitKey(1) == ord('q'):
                device.release()
                exit()

    # Function to initialize/reset tracking parameters
    def reset_tracking_parameters():
        """Reset all tracking parameters to initial state for a new tracking session or after contact loss."""
        return (
            reference.C.copy(),   # C_ref: Contact mask of current reference frame
            reference.H.copy(),   # H_ref: Height map of current reference frame
            reference.N.copy(),   # N_ref: Normal map of current reference frame
            None,                 # C_prev: Contact mask of previous frame
            None,                 # H_prev: Height map of previous frame
            None,                 # N_prev: Normal map of previous frame
            np.eye(4),            # prev_T_ref: Transformation from previous frame to current reference frame
            np.eye(4),            # start_T_ref: Cumulative transformation from the *initial* reference frame to current reference frame
            False,                # has_contact_history: Flag to check if there was previous contact
        )

    # Initialize tracking variables
    C_ref, H_ref, N_ref, C_prev, H_prev, N_prev, \
    prev_T_ref, start_T_ref, has_contact_history = reset_tracking_parameters()

    # Add flags for angle tracking
    angle_offset = 0.0
    first_frame_angle_adjusted = False
    first_frame_after_contact_established = True

    # Real-time object tracking loop
    print("\nStarting real-time object tracking with automatic reference detection. Press any key to quit.\n")

    is_running = True
    while is_running:
        image_curr = device.get_image()
        if image_curr is None:
            print("Warning: Could not get image from device. Skipping frame.")
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue

        G_curr, H_curr, C_curr = recon.get_surface_info(image_curr, ppmm)
        C_curr = erode_contact_mask(C_curr)

        # No sufficient contact in current frame
        if np.sum(C_curr) < MIN_CONTACT_AREA_PIXELS:
            
            if has_contact_history: #clean origin image
                print(f"Contact lost (area < {MIN_CONTACT_AREA_PIXELS} pixels) - resetting tracking parameters.")
                has_contact_history = False
                # 重新生成参考图像
                print("等待重新接触...")
                reference = None
                while True:
                    new_frame = device.get_image()
                    if new_frame is None:
                        continue
                    
                    reference = generate_reference_from_first_frame(new_frame, recon, ppmm, MIN_CONTACT_AREA_PIXELS)
                    if reference is not None:
                        print("重新生成参考图像成功！")
                        C_ref, H_ref, N_ref, C_prev, H_prev, N_prev, \
                        prev_T_ref, start_T_ref, has_contact_history = reset_tracking_parameters()
                        angle_offset = 0.0
                        first_frame_angle_adjusted = False
                        break
                    else:
                        # 只显示左侧图像，显示"waiting for contact"
                        display_image = np.zeros((imgh, imgw * 2, 3), dtype=np.uint8)
                        cv2.putText(
                            display_image,
                            "Waiting for Contact",
                            (imgw // 2, imgh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow("frame", display_image)
                        if cv2.waitKey(1) != -1:
                            device.release()
                            exit()
                continue
            display_image = np.zeros((imgh, imgw * 2, 3), dtype=np.uint8)
            cv2.putText(
                display_image,
                "Waiting for Contact",
                (imgw // 2, imgh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.imshow("frame", display_image)
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue

        # Mark that we now have contact
        has_contact_history = True
        N_curr = gxy2normal(G_curr)

        # Calculate current frame's centroid directly from contact mask
        cx_curr, cy_curr = calculate_contact_centroid(C_curr)
        if cx_curr is None or cy_curr is None:
            print("Warning: Could not calculate centroid of current frame. Skipping frame.")
            cv2.imshow("frame", image_curr)
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue

        curr_T_ref = None
        is_reset_due_to_error = False
        max_iteration_tag = False

        try:
            # Attempt to track from current reference frame
            curr_T_ref, max_iteration_tag = normalflow(
                N_ref,
                C_ref,
                H_ref,
                N_curr,
                C_curr,
                H_curr,
                prev_T_ref,
                ppmm,
                5000,
                verbose=True
            )
        except InsufficientOverlapError:
            print("Insufficient overlap with current reference frame. Attempting reset.")
            is_reset_due_to_error = True
            if N_prev is None: 
                continue

        # Skip first frame after contact re-establishment if iteration count is bad
        if first_frame_after_contact_established:
            if max_iteration_tag == True:
                print(f"Skipping first frame after contact re-establishment due to bad iteration count.")
                has_contact_history = False
                cv2.imshow("frame", image_curr)
                key = cv2.waitKey(1)
                if key != -1:
                    is_running = False
                continue

        # Reset reference frame (set a new keyframe) if needed
        if N_prev is not None: # not first frame
            curr_T_prev = None
            try:
                # Calculate transformation from previous frame to current frame
                curr_T_prev, _ = normalflow(
                    N_prev,
                    C_prev,
                    H_prev,
                    N_curr,
                    C_curr,
                    H_curr,
                    np.eye(4),
                    ppmm,
                )
            except InsufficientOverlapError:
                print("Reset failed: Previous frame also has insufficient overlap with current. Skipping frame.")
                cv2.imshow("frame", image_curr)
                key = cv2.waitKey(1)
                if key != -1:
                    is_running = False
                continue

            # If curr_T_ref was successfully calculated, check for large errors against curr_T_prev
            if curr_T_ref is not None and not is_reset_due_to_error:
                T_error = np.linalg.inv(curr_T_ref) @ curr_T_prev @ prev_T_ref
                pose_error = transform2pose(T_error)
                rot_error = np.linalg.norm(pose_error[3:])
                trans_error = np.linalg.norm(pose_error[:3])

                if rot_error > RESET_ROT_THRESHOLD or trans_error > RESET_TRANS_THRESHOLD:
                    print(f"Large pose error detected (Rot: {rot_error:.2f} deg, Trans: {trans_error:.2f} m). Resetting reference frame.")
                    is_reset_due_to_error = True
            
            # If a reset is triggered
            if is_reset_due_to_error or is_reset:
                # Reset to previous frame as the new reference (keyframe)
                print("Resetting reference frame to previous frame.")
                C_ref = C_prev.copy()
                H_ref = H_prev.copy()
                N_ref = N_prev.copy()
                start_T_ref = start_T_ref @ np.linalg.inv(prev_T_ref)
                curr_T_ref = curr_T_prev.copy()
                is_reset = False

        if curr_T_ref is None:
            print("Warning: Could not establish valid transformation for current frame. Skipping.")
            cv2.imshow("frame", image_curr)
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue

        # Update states for the next iteration
        C_prev = C_curr.copy()
        H_prev = H_curr.copy()
        N_prev = N_curr.copy()
        prev_T_ref = curr_T_ref.copy()

        # --- Display the object tracking result ---

        # 显示参考图像
        image_l = reference.image.copy()
        cv2.putText(
            image_l,
            "Reference Frame",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        center_start = np.array([reference.cx, reference.cy]).astype(np.int32)
        unit_vectors_start = reference.get_coordinate_system_vectors()
        annotate_coordinate_system(image_l, center_start, unit_vectors_start)

        # 显示当前帧
        image_r = image_curr.copy()
        cv2.putText(
            image_r,
            "Current Frame (Tracking)",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        center_curr = np.array([cx_curr, cy_curr]).astype(np.int32)

        # Calculate cumulative transformation
        curr_T_start = curr_T_ref @ np.linalg.inv(start_T_ref)

        # Extract rotation angle from transformation matrix
        R_curr = curr_T_start[:3, :3]
        raw_angle_z_rad = np.arctan2(R_curr[1, 0], R_curr[0, 0])

        # 计算相对于参考坐标系的角度
        calculated_angle = raw_angle_z_rad + reference.principal_angle

        # 只在第一帧调整角度到右半平面
        if first_frame_after_contact_established and not first_frame_angle_adjusted:
            adjusted_first_angle = adjust_angle_to_right_half(calculated_angle)
            angle_offset = adjusted_first_angle - calculated_angle
            angle_z_rad = adjusted_first_angle
            first_frame_after_contact_established = False
            first_frame_angle_adjusted = True
            print(f"First frame angle adjustment: raw={np.degrees(calculated_angle):.2f}°, adjusted={np.degrees(angle_z_rad):.2f}°, offset={np.degrees(angle_offset):.2f}°")
        else:
            # 对于后续帧，应用相同的偏移量以保持连续性
            angle_z_rad = calculated_angle + angle_offset

        angle_z_deg = np.degrees(angle_z_rad)

        # 获取用于可视化的坐标系向量
        coord_vectors = get_coordinate_vectors_from_angle(angle_z_rad, scale=1)
        annotate_coordinate_system(image_r, center_curr, coord_vectors)

        # 计算在当前参考坐标系中的位移
        center_3d_curr = (
            np.array(
                [(cx_curr - imgw / 2 + 0.5), (cy_curr - imgh / 2 + 0.5), 0]
            )
            * ppmm / 1000.0
        )

        point = center_3d_curr[:2]

        # 应用参考坐标系的变换
        T_ref = reference.get_coordinate_transform_matrix()
        point_transformed = (T_ref[:2, :2] @ point.reshape(-1, 1)).flatten()

        # 计算直线方程
        if np.abs(np.cos(angle_z_rad)) < 1e-6:
            print(f"直线在图像坐标系下为竖线: x = {point_transformed[0]:.6f}")
            # 对于竖线，设置一个很大的截距值来表示垂直线
            b = float('inf') if point_transformed[0] >= 0 else float('-inf')
        else:
            k = np.tan(angle_z_rad)
            b = point_transformed[1] - k * point_transformed[0]
            # print(f"图像坐标系 - 斜率 k = {k:.6f}, 截距 b = {b:.6f}, 角度 = {angle_z_deg:.6f}°")

        # publish degree via ROS, publisher is ros_pub_tracking
        publish_tracking_data(ros_pub_tracking, angle_z_deg, b, center_3d_curr[0], center_3d_curr[1])

        # 在图像上显示角度信息
        cv2.putText(
            image_r,
            f"Angle (angle_z_deg): {angle_z_deg:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Display combined image (左边reference，右边current)
        cv2.imshow("frame", cv2.hconcat([image_l, image_r]))

        # Publish image_r via ROS
        print(f"[DEBUG] 准备发布图像，尺寸: {image_r.shape}")
        publish_image(ros_pub_image, ros_bridge, image_r)
        
        # Publish raw image via ROS (without any annotations)
        print(f"[DEBUG] 准备发布原始图像，尺寸: {image_curr.shape}")
        publish_raw_image(ros_pub_raw_image, ros_bridge, image_curr)

        key = cv2.waitKey(1)
        if key == ord('q'):
            is_running = False
        elif key == ord('r'):
            reference = generate_reference_from_first_frame(image_curr, recon, ppmm, MIN_CONTACT_AREA_PIXELS)
        elif key == ord("i"):
            collect_background_images(device, recon)

    device.release()
    cv2.destroyAllWindows()
    print("Streaming session ended.")


if __name__ == "__main__":
    realtime_object_tracking()
