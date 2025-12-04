#!/usr/bin/env python3
"""
测试交互式坐标选择功能
"""
from create_camera import CreateRealsense
import cv2

if __name__ == "__main__":
    # 初始化相机（使用您的相机序列号）
    camera = CreateRealsense("231522072272")
    
    print("相机初始化完成，准备进行交互式坐标选择...")
    print("=" * 60)
    
    # 调用交互式坐标选择功能
    point_info = camera.get_point_coordinate(window_name="测试点选择")
    
    if point_info is not None:
        print("\n✅ 成功获取点的坐标信息:")
        print(f"  像素坐标: {point_info['pixel']}")
        print(f"  相机坐标系 (X, Y, Z): {point_info['camera_coord']} m")
        print(f"  深度值: {point_info['depth']:.4f} m")
    else:
        print("\n❌ 未获取到坐标信息")
    
    # 释放相机资源
    camera.release()
    cv2.destroyAllWindows()
    print("\n测试完成，相机资源已释放")


