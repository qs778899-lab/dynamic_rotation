"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""


import time
import json
from spatialmath import SE3, SO3
import numpy as np



from simple_api import SimpleApi
from dobot_gripper import DobotGripper
from create_camera import CreateRealsense




class create_env:
    def __init__(self, config_pth, init_robot=True, camera_id="camera_1"):
        """
        初始化环境
        
        Args:
            config_pth: 配置文件路径
            init_robot: 是否在初始化时自动初始化机器人，默认为True
                       - True: 自动调用 init_robot1() 初始化机器人和夹爪
                       - False: 不初始化机器人，可以稍后手动调用 init_dobot()
            camera_id: 相机配置ID，默认为"camera_1"，可选"camera_3"等
        """
        with open(config_pth, 'r') as file:
            self.config = json.load(file)

        # 根据参数决定是否自动初始化
        if init_robot:
            self.robot1, self.gripper = self.init_robot1()
        else:
            self.robot1 = None
            self.gripper = None
        
        self.camera1_main = self.init_camera(camera_id)



    def init_camera(self, camera_id="camera_1"):
        """
        初始化相机
        
        Args:
            camera_id: 相机配置ID，默认为"camera_1"
        
        Returns:
            camera_main: 包含相机对象、变换矩阵和内参的字典
        """
        camera_cfg = self.config[camera_id]
        camera = CreateRealsense(camera_cfg['id'])
        camera_rt = SE3.Rt(
            np.array(camera_cfg['T_ee_cam']['rotation_matrix']),
            camera_cfg['T_ee_cam']['translation_vector'],
            check=False
        )
        camera_main = {
            "cam": camera,
            "Rt": camera_rt,
            "T_ee_cam": camera_rt,
            "cam_k": np.array(camera_cfg['cam_k'])
        }
        self.camera1 = camera
        self.camera1_rt = camera_rt
        return camera_main
                    

    def init_robot1(self):
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
        robot_main = {
            "robot": dobot,
            "tcp2ee": self.config['robot1']['tcp2ee']
        }
        return robot_main,gripper

    def init_dobot(self):
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
        return dobot