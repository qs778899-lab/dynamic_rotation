# Manipulation framework based on Robotwin API

## 项目结构

config.json: cam, dobot等配置参数

GraspLibrary.json: target object的抓取配置参数

env.py: 软硬件对象初始化

level2_action.py: level2 actions 定义

example main, eg: glassbar_grasp_main_smootharray_dotarray.py


## 不同运行主文件的功能:

test_glassbar_grasp_main_tt_contact.py: 测试玻璃棒向下移动碰到漏斗壁是否及时停止

glassbar_grasp_main_tt_contact.py: 抓取玻璃棒调整姿态至倾斜向下，再向下移动触碰桌面

glassbar_grasp_main_v_contact.py: 抓取玻璃棒调整姿态至垂直向下，再向下移动触碰桌面

test_changes_in_YIMU_monitor.py: 测试不同id对应的视触觉传感器设备

compare_force_monitor.py: 对比力控和视触觉传感器对微小力的感知灵敏度

