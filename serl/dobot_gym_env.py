"""
Gym environment wrapping Dobot arm and cameras for SERL.

Matches the serl_robot_infra.franka_env.FrankaEnv contract:
- observation_space: Dict with "state" and image keys
- action_space: 7D Box in [-1, 1]: [dx, dy, dz, droll, dpitch, dyaw, gripper]
- step/reset return (obs, reward, done, truncated, info)

Notes:
- This env is an HTTP client: it sends commands to the Dobot Flask server
  (serl_dobot/robot_servers/dobot_server.py) and polls state via POST.
- Cameras can be local USB (CameraReader); if unavailable, zeros are returned.
"""

import copy
import time
from typing import Dict, Optional, Tuple

import cv2
import gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from serl_dobot.camera_reader import CameraReader


class DobotEnvConfig:
    """Configuration for DobotEnv (HTTP client to Dobot server)."""

    SERVER_URL: str = "http://127.0.0.1:5001/"

    # Action scaling
    ACTION_SCALE = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 500.0], dtype=np.float32)

    # Safety limits (absolute Cartesian bounds in mm)
    XYZ_LIMIT_LOW = np.array([-300.0, -300.0, 0.0])
    XYZ_LIMIT_HIGH = np.array([300.0, 300.0, 400.0])

    # Episode length
    MAX_EPISODE_LENGTH = 100

    # Camera
    CAMERA_ID = 0  # USB cam id
    IMAGE_SHAPE = (128, 128, 3)  # H, W, C


class DobotEnv(gym.Env):
    """
    HTTP client env that mirrors serl_robot_infra.franka_env.FrankaEnv:
    - step() sends pose/gripper commands via HTTP POST to Dobot server
    - getstate via HTTP POST
    - camera captured locally (optional)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        config: DobotEnvConfig = DobotEnvConfig(),
    ):
        super().__init__()
        self.cfg = config
        self.hz = hz
        self.dt = 1.0 / hz
        self.fake_env = fake_env

        # Action: 7D in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.ones((7,), dtype=np.float32) * -1.0,
            high=np.ones((7,), dtype=np.float32),
        )

        # Observation space
        self.image_keys = ("front",)
        image_space = {
            k: gym.spaces.Box(
                low=0,
                high=255,
                shape=self.cfg.IMAGE_SHAPE,
                dtype=np.uint8,
            )
            for k in self.image_keys
        }
        state_space = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                **image_space,
            }
        )

        # Runtime vars
        self.step_count = 0
        self.curr_pose = np.zeros(7, dtype=np.float32)
        self.curr_vel = np.zeros(6, dtype=np.float32)
        self.curr_force = np.zeros(3, dtype=np.float32)
        self.curr_torque = np.zeros(3, dtype=np.float32)
        self.curr_gripper = np.zeros(1, dtype=np.float32)

        self.camera = None
        if not self.fake_env:
            self.camera = CameraReader(camera_id=self.cfg.CAMERA_ID, init_camera=True)
            self._update_state()

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        if not self.fake_env:
            self._recover()
            self._go_home()
            self._update_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_count += 1

        if not self.fake_env:
            self._apply_action(action)
            time.sleep(self.dt)
            self._update_state()

        obs = self._get_obs()
        reward = 0.0  # placeholder sparse reward
        done = self.step_count >= self.cfg.MAX_EPISODE_LENGTH
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _apply_action(self, action: np.ndarray):
        # action: [dx, dy, dz, droll, dpitch, dyaw, grip] in [-1,1]
        delta = action * self.cfg.ACTION_SCALE
        # current pose is xyz + quat; convert quat to euler, add delta rot, convert back
        curr_xyz = self.curr_pose[:3]
        curr_quat = self.curr_pose[3:]
        curr_euler = Rotation.from_quat(curr_quat).as_euler("xyz")

        target_xyz = curr_xyz + delta[:3]
        target_xyz = np.clip(target_xyz, self.cfg.XYZ_LIMIT_LOW, self.cfg.XYZ_LIMIT_HIGH)
        target_euler = curr_euler + np.deg2rad(delta[3:6])
        target_quat = Rotation.from_euler("xyz", target_euler).as_quat()
        target_pose = np.concatenate([target_xyz, target_quat]).astype(np.float32)

        grip = float(np.clip((action[6] * 0.5 + 0.5) * self.cfg.ACTION_SCALE[-1], 0, 1000))

        # Send HTTP commands
        self._send_pos_command(target_pose)
        self._send_gripper_command(grip)

    def _send_pos_command(self, pose: np.ndarray):
        data = {"arr": pose.tolist()}
        requests.post(self.cfg.SERVER_URL + "pose", json=data)

    def _send_gripper_command(self, pos: float):
        data = {"gripper_pos": pos}
        requests.post(self.cfg.SERVER_URL + "move_gripper", json=data)

    def _recover(self):
        try:
            requests.post(self.cfg.SERVER_URL + "clearerr")
        except Exception:
            pass

    def _go_home(self):
        requests.post(self.cfg.SERVER_URL + "home")

    def _update_state(self):
        ps = requests.post(self.cfg.SERVER_URL + "getstate").json()
        self.curr_pose[:] = np.array(ps["pose"], dtype=np.float32)
        self.curr_vel[:] = np.array(ps["vel"], dtype=np.float32)
        self.curr_force[:] = np.array(ps["force"], dtype=np.float32)
        self.curr_torque[:] = np.array(ps["torque"], dtype=np.float32)
        self.curr_gripper[:] = np.array(ps["gripper_pos"], dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        images = {}
        for k in self.image_keys:
            img = self._get_image()
            images[k] = img

        state = {
            "tcp_pose": self.curr_pose,
            "tcp_vel": self.curr_vel,
            "gripper_pose": self.curr_gripper,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
        }
        return copy.deepcopy({**images, "state": state})

    def _get_image(self):
        if self.camera is None or self.fake_env:
            return np.zeros(self.cfg.IMAGE_SHAPE, dtype=np.uint8)
        frame = self.camera.get_current_frame()
        if frame is None:
            return np.zeros(self.cfg.IMAGE_SHAPE, dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.cfg.IMAGE_SHAPE[1], self.cfg.IMAGE_SHAPE[0]))
        return frame.astype(np.uint8)


__all__ = ["DobotEnv", "DobotEnvConfig"]

