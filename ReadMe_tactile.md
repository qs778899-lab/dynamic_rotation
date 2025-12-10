# 视触觉传感器使用说明

## 运行流程

1. 
cd /home/erlin/work/tasks
conda activate normalflow
bash create_virtualcam.bash
或者需要创建多个虚拟相机：
bash create_virtualcam_multi.bash


2. 
在终端终结者运行roscore

3. 
cd /home/erlin/work/tasks
conda activate normalflow
bash open_sn.bash
实时看到传感器画面

4. 计算object_pose (二维的pose)
open_sn.bash中运行的realtime_object_tracking.py会计算得到object pose, 也会发送ros topic

5. 测试是否已经发送topic(ros node和ros topic不同)

可以查看ros通信发送情况：
rqt

查看所有活跃的节点：
rosnode list

查看所有活跃的话题（存在但不一定有实际数据发出）：
rostopic list
（注意，当没有contact时，tracking_data没有数据会发出）


查看具体某个ros topic是否有数据流：
rostopic echo -p /tracking_data 
rostopic echo -p /image_object_orientation 


查看哪些node在发布the topic消息:
rostopic info tracking_data




6. 测试能否接收ros topic和print出正确的结果
运行test_ros_subscriber.py 测试文件



