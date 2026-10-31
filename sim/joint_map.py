"""Convert joint values between the real follower (LeRobot calibrated units) and the sim (rad).

Real: body joints in degrees (zero = middle of the calibrated range), gripper 0 (closed) to 100 (open).
Sim:  radians, joint order JOINTS. Offsets and gripper endpoints come from config `sim_map`.
"""

from pathlib import Path

import numpy as np
import yaml

from sim.env import JOINTS


class JointMap:
    def __init__(self, offsets_deg: dict[str, float], gripper_sim_deg: tuple[float, float]):
        self.offsets = np.radians([offsets_deg[j] for j in JOINTS[:5]])
        self.g_closed, self.g_open = np.radians(gripper_sim_deg)

    @classmethod
    def from_config(cls, path: str | Path) -> "JointMap":
        cfg = yaml.safe_load(Path(path).read_text())["sim_map"]
        return cls(cfg["offsets_deg"], tuple(cfg["gripper_sim_deg"]))

    def real_to_sim(self, real: np.ndarray) -> np.ndarray:
        """(..., 6) real values -> (..., 6) sim radians."""
        real = np.asarray(real, dtype=np.float64)
        body = np.radians(real[..., :5]) - self.offsets
        gripper = self.g_closed + real[..., 5:] / 100 * (self.g_open - self.g_closed)
        return np.concatenate([body, gripper], axis=-1)

    def sim_to_real(self, sim: np.ndarray) -> np.ndarray:
        """(..., 6) sim radians -> (..., 6) real values."""
        sim = np.asarray(sim, dtype=np.float64)
        body = np.degrees(sim[..., :5] + self.offsets)
        gripper = (sim[..., 5:] - self.g_closed) / (self.g_open - self.g_closed) * 100
        return np.concatenate([body, gripper], axis=-1)
