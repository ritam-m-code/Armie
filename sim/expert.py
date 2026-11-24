"""Scripted pick expert for PickEnv, used to generate sim demonstrations.

Uses privileged state (the object's true pose) and the mink IK library on an arm-only copy of the scene.
Plans the whole episode open-loop at reset: home -> pre-grasp (above) -> straight-line descent -> close -> lift.

Grasp geometry (site frame of `gripperframe`): x = approach (points out of the fingertips), z = closing
direction (from the fixed finger toward the moving jaw). The site sits on the fixed finger's inner face at its tip.
"""

import mink
import mujoco
import numpy as np

from sim.env import SCENE, PickEnv

GRASP_INSET = 0.017  # m from fixed-finger face to object centre along the closing axis (cube half-width + 2 mm)
GRIPPER_OPEN = 0.7  # rad
GRIPPER_CLOSED = -0.17  # rad, sim lower limit = real 0
# Pointing straight down, the wrist_flex limit caps the reachable height at ~5 cm above the object,
# so the lift lets the gripper tilt (low orientation cost).
PRE_GRASP_HEIGHT = (0.03, 0.045)  # m above grasp point
LIFT_HEIGHT = 0.10  # m


def _arm_only_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.delete(spec.body("object"))
    return spec.compile()


def _grasp_rotation(closing_yaw: float) -> np.ndarray:
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(closing_yaw), np.sin(closing_yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def _min_jerk(n: int) -> np.ndarray:
    s = np.linspace(0, 1, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class PickExpert:
    def __init__(self, env: PickEnv):
        self.env = env
        model = _arm_only_model()
        self._cfg = mink.Configuration(model)
        self._task = mink.FrameTask("gripperframe", "site", position_cost=1.0, orientation_cost=0.3, lm_damping=1e-2)
        self._limits = [mink.ConfigurationLimit(model)]

    def _ik(self, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, orientation_cost: float = 0.3, iters: int = 400):
        self._cfg.update(q0.copy())
        self._task.set_orientation_cost(orientation_cost)
        self._task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos))
        for _ in range(iters):
            vel = mink.solve_ik(self._cfg, [self._task], 0.1, "daqp", 1e-3, limits=self._limits)
            self._cfg.integrate_inplace(vel, 0.1)
            err = self._task.compute_error(self._cfg)
            if np.linalg.norm(err[:3]) < 5e-4 and np.linalg.norm(err[3:]) < 1e-2:
                break
        return self._cfg.q.copy(), float(np.linalg.norm(err[:3]))

    def _line(self, q, p0, p1, rot, n, orientation_cost=0.3):
        qs = []
        for s in np.linspace(0, 1, n + 1)[1:]:
            q, _ = self._ik(q, p0 + (p1 - p0) * s, rot, orientation_cost, iters=150)
            qs.append(q)
        return qs

    def plan(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Joint-target trajectory (T, 6) in sim rad for the current env state, and the grasp IK error (m)."""
        data = self.env.data
        obj_pos = data.body("object").xpos.copy()
        obj_rot = data.body("object").xmat.reshape(3, 3)

        # Close across a pair of cube faces. Try the 4 face normals in random order (so both finger assignments
        # appear in the data) and keep the first whose grasp IK converges; some need wrist_roll past its limit.
        face_yaw = np.arctan2(obj_rot[1, 0], obj_rot[0, 0])
        jitter = np.r_[rng.uniform(-0.003, 0.003, 2), 0.0]
        q_home = data.qpos[:6].copy()
        best = None
        for k in rng.permutation(4):
            rot = _grasp_rotation(face_yaw + k * np.pi / 2)
            grasp = obj_pos - GRASP_INSET * rot[:, 2] + jitter
            q_grasp, grasp_err = self._ik(q_home, grasp, rot)
            if best is None or grasp_err < best[3]:
                best = (rot, grasp, q_grasp, grasp_err)
            if grasp_err < 1e-3:
                break
        rot, grasp, q_grasp, grasp_err = best

        above = grasp + [0, 0, rng.uniform(*PRE_GRASP_HEIGHT)]
        lift = grasp + [0, 0, LIFT_HEIGHT]
        q_above, _ = self._ik(q_grasp, above, rot)

        n = lambda base: int(base * rng.uniform(0.8, 1.25))  # noqa: E731
        traj = []
        open_g = np.r_[q_above[:5], GRIPPER_OPEN]
        traj += [q_home + (open_g - q_home) * s for s in _min_jerk(n(45))]
        traj += [np.r_[q[:5], GRIPPER_OPEN] for q in self._line(q_above, above, grasp, rot, n(25))]
        q_closed = np.r_[traj[-1][:5], GRIPPER_CLOSED]
        traj += [traj[-1] + (q_closed - traj[-1]) * s for s in _min_jerk(n(12))]
        traj += [q_closed] * 5
        traj += [np.r_[q[:5], GRIPPER_CLOSED] for q in self._line(q_closed, grasp, lift, rot, n(30), 0.05)]
        traj += [traj[-1]] * 15
        return np.array(traj), grasp_err
