"""Gymnasium env for the SO-101 pick scene.

Observations follow LeRobot's sim-env convention so episodes export directly to a LeRobot dataset:
    agent_pos: (6,) joint positions in rad, order JOINTS
    pixels:    {"front": HxWx3 uint8, "wrist": HxWx3 uint8}
Action: (6,) joint position targets in rad, clipped to each actuator's ctrlrange.
Joint zeros are at mid-range ("new calibration"), matching LeRobot's calibrated convention.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent / "assets" / "scene.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = ["front", "wrist"]

# Object spawn region on the table, in the base frame (m). TODO(verify) against the real reachable workspace.
SPAWN_X = (0.15, 0.30)
SPAWN_Y = (-0.12, 0.12)
LIFT_SUCCESS_M = 0.05


class PickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, fps: int = 30, image_size: tuple[int, int] = (480, 640), render_mode: str = "rgb_array"):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.n_substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.dt = self.n_substeps * self.model.opt.timestep

        self._qpos_idx = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self._act_idx = np.array([self.model.actuator(j).id for j in JOINTS])
        self._obj_qpos = self.model.joint("object").qposadr[0]
        self._obj_body = self.model.body("object").id
        ctrl_range = self.model.actuator_ctrlrange[self._act_idx]

        h, w = image_size
        self._renderer = mujoco.Renderer(self.model, h, w)
        self.action_space = gym.spaces.Box(ctrl_range[:, 0], ctrl_range[:, 1], dtype=np.float64)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (len(JOINTS),), dtype=np.float64),
                "pixels": gym.spaces.Dict(
                    {cam: gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8) for cam in CAMERAS}
                ),
            }
        )
        self._obj_z0 = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        x, y = self.np_random.uniform(*SPAWN_X), self.np_random.uniform(*SPAWN_Y)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        half_height = self.model.geom("object").size[2]
        self.data.qpos[self._obj_qpos : self._obj_qpos + 7] = [x, y, half_height, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.ctrl[self._act_idx] = self.data.qpos[self._qpos_idx]

        mujoco.mj_forward(self.model, self.data)
        self._obj_z0 = self.data.xpos[self._obj_body, 2]
        return self._obs(), {"is_success": False}

    def step(self, action):
        self.data.ctrl[self._act_idx] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        success = bool(self.data.xpos[self._obj_body, 2] - self._obj_z0 > LIFT_SUCCESS_M)
        return self._obs(), float(success), success, False, {"is_success": success}

    def render(self):
        return self._render_camera("front")

    def close(self):
        self._renderer.close()

    def _obs(self):
        return {
            "agent_pos": self.data.qpos[self._qpos_idx].copy(),
            "pixels": {cam: self._render_camera(cam) for cam in CAMERAS},
        }

    def _render_camera(self, cam: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()
