
# 项目结构:

1. serl_launcher (high-level RL policy):

    action维度自适应，由env如franka_env定义

    RL架构是分布式RL架构，learner node 和 actor node之间通信基于agentlace

2. robot_servers (bridge between high-level rl policy and low-level robot arm controller):

    通信方式: HTTP POST, 不依赖ROS环境，有利于RL库环境和ROS依赖的隔离

    ***_env.py(如franka_env.py): send http post

    ***_server.py(如franka_server.py): receive http post

3. 





# 真机训练流程(以franka arm为例):

    # Terminal 0: 启动机器人服务器 (必须先启动)
    python serl_robot_infra/robot_servers/franka_server.py \
        --robot_ip=172.16.0.2 \
        --gripper_type=Robotiq

    # Terminal 1: 启动 Learner (可以在 GPU 服务器上)
    bash run_learner.sh

    # Terminal 2: 启动 Actor (必须在能连接机器人的电脑上)
    bash run_actor.sh