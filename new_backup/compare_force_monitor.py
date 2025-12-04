import cv2
import time
import rospy
from camera_reader import CameraReader
from env import create_env


def main():
    env = create_env("config.json", init_robot=False)  # init_robot=False 表示不在创建 env 时自动初始化机器人
    dobot = env.init_dobot()

    camera = CameraReader(camera_id=11) #! 注意id有时会变化

    pixel_threshold = 2
    min_area = 2
    change_threshold = 3
    window_name = "Camera Preview"
    save_dir = None

    rospy.init_node('test', anonymous=True) 

    if camera.cap is None or not camera.cap.isOpened():
        print("无法启动相机，退出实时监测。")
        return

    prev_frame = camera.get_current_frame()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    step = 0
    while True:
        wait = rospy.Rate(10)  
        wait.sleep()
        frame = camera.get_current_frame()
        cv2.imshow(window_name, frame)

        # 实时获取力传感器数值
        force_values = dobot.get_force()
        if force_values and len(force_values) == 6:
            fx, fy, fz, frx, fry, frz = force_values
            max_force_component = max(abs(value) for value in force_values)
            print(f"[力传感器] Fx:{fx:6.2f} Fy:{fy:6.2f} Fz:{fz:6.2f} | "
                  f"FRx:{frx:6.2f} FRy:{fry:6.2f} FRz:{frz:6.2f} | Max:{max_force_component:6.2f}N")
        else:
            print("[力传感器] 读取失败或数据异常")

        result = camera.detect_pixel_changes(
            prev_frame,
            frame,
            threshold=pixel_threshold,
            min_area=min_area,
            save_dir=save_dir,
            step_num=step
        )

        if 'error' in result:
            print(f"[变化检测] 第{step}帧检测出错: {result['error']}")
        else:
            if result['change_percentage'] >= change_threshold:
                print(
                    f"[变化检测] 第{step}帧检测到变化: "
                    f"{result['change_percentage']:.2f}% "
                    f"({result['changed_pixels']} px, {result['num_changes']} 区域)"
                )
            else:
                print(
                    f"[变化检测] 第{step}帧无显著变化 "
                    # f"({result['change_percentage']:.2f}% < {change_threshold}%)"
                )

        prev_frame = frame
        step += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("检测结束，用户主动退出。")
            break

        time.sleep(0.01)

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()

