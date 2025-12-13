#!/usr/bin/env python3
"""
Flask server bridging Dobot API to HTTP, mirroring franka_server endpoints.

Endpoints:
- /pose             : POST {"arr": [x, y, z, qx, qy, qz, qw]}
- /getstate         : POST -> {pose, vel, force, torque, q, dq, jacobian, gripper_pos}
- /move_gripper     : POST {"gripper_pos": float in [0,1000]}
- /open_gripper     : POST
- /close_gripper    : POST
- /clearerr         : POST
- /home             : POST

Note: Dobot API does not provide impedance control; this server offers position control.
"""

import argparse
import time
import numpy as np
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R

from serl_dobot.simple_api import SimpleApi
from serl_dobot.dobot_gripper import DobotGripper


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, default="192.168.5.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--home", nargs=6, type=float, default=[250, 0, 150, 0, 0, 0])
    return parser


class DobotServer:
    def __init__(self, robot_ip: str, home_pose):
        self.robot_ip = robot_ip
        self.home_pose_rpy = np.array(home_pose, dtype=np.float32)  # x,y,z,r,p,y (deg)

        self.api = SimpleApi(robot_ip, 29999)
        self.api.clear_error()
        self.api.enable_robot()
        self.api.stop()
        self.api.enable_ft_sensor(1)
        time.sleep(1.0)
        self.api.six_force_home()
        time.sleep(1.0)

        self.gripper = DobotGripper(self.api)
        self.gripper.connect(init=True)

        # state placeholders
        self.pos = np.zeros(7, dtype=np.float32)  # xyz + quat
        self.vel = np.zeros(6, dtype=np.float32)
        self.force = np.zeros(3, dtype=np.float32)
        self.torque = np.zeros(3, dtype=np.float32)
        self.q = np.zeros(6, dtype=np.float32)
        self.dq = np.zeros(6, dtype=np.float32)
        self.jacobian = np.zeros((6, 6), dtype=np.float32)
        self.gripper_pos = 0.0

        # go home once
        self.move_home()
        self._update_state()

    # ------------------------ Robot ops ------------------------ #
    def move_pose_quat(self, pose_quat: np.ndarray):
        # Convert quat to rpy (deg) for Dobot
        xyz = pose_quat[:3]
        quat = pose_quat[3:]
        rpy_rad = R.from_quat(quat).as_euler("xyz")
        rpy_deg = np.rad2deg(rpy_rad)
        self.api.move_to_pose(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(rpy_deg[0]),
            float(rpy_deg[1]),
            float(rpy_deg[2]),
            speed=30,
            acceleration=3,
        )

    def move_home(self):
        h = self.home_pose_rpy
        self.api.move_to_pose(
            float(h[0]),
            float(h[1]),
            float(h[2]),
            float(h[3]),
            float(h[4]),
            float(h[5]),
            speed=50,
            acceleration=5,
        )
        self.gripper.control(position=800, force=50, speed=50)
        self._update_state()

    def _update_state(self):
        pose_rpy = np.array(self.api.get_pose(), dtype=np.float32)
        quat = R.from_euler("xyz", np.deg2rad(pose_rpy[3:])).as_quat()
        self.pos = np.concatenate([pose_rpy[:3], quat]).astype(np.float32)
        self.vel = np.zeros(6, dtype=np.float32)
        self.force = np.zeros(3, dtype=np.float32)
        self.torque = np.zeros(3, dtype=np.float32)
        self.q = np.zeros(6, dtype=np.float32)
        self.dq = np.zeros(6, dtype=np.float32)
        self.jacobian = np.zeros((6, 6), dtype=np.float32)
        gp = self.gripper.read_current_position()
        if gp is not None and len(gp) > 0:
            self.gripper_pos = float(gp[0]) / 1000.0
        else:
            self.gripper_pos = 0.0

    # ------------------------ Gripper ops ---------------------- #
    def open_gripper(self):
        self.gripper.control(position=900, force=80, speed=80)
        self.gripper_pos = 0.9

    def close_gripper(self):
        self.gripper.control(position=100, force=80, speed=80)
        self.gripper_pos = 0.1

    def move_gripper(self, pos):
        # pos in [0,1000]
        pos = float(np.clip(pos, 0, 1000))
        self.gripper.control(position=pos, force=80, speed=80)
        self.gripper_pos = pos / 1000.0

    def clear(self):
        self.api.clear_error()


def create_app(robot_ip: str, home_pose):
    server = DobotServer(robot_ip, home_pose)
    app = Flask(__name__)

    @app.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        server.move_pose_quat(pos)
        server._update_state()
        return "Moved"

    @app.route("/getstate", methods=["POST"])
    def get_state():
        server._update_state()
        return jsonify(
            {
                "pose": server.pos.tolist(),
                "vel": server.vel.tolist(),
                "force": server.force.tolist(),
                "torque": server.torque.tolist(),
                "q": server.q.tolist(),
                "dq": server.dq.tolist(),
                "jacobian": server.jacobian.tolist(),
                "gripper_pos": server.gripper_pos,
            }
        )

    @app.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = float(request.json.get("gripper_pos", 500))
        server.move_gripper(pos)
        return "Moved"

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        server.open_gripper()
        return "Opened"

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        server.close_gripper()
        return "Closed"

    @app.route("/clearerr", methods=["POST"])
    def clearerr():
        server.clear()
        return "Cleared"

    @app.route("/home", methods=["POST"])
    def home():
        server.move_home()
        return "Homed"

    return app


def main():
    parser = build_parser()
    args = parser.parse_args()
    app = create_app(args.robot_ip, args.home)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

