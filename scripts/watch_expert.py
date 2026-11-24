"""Watch the scripted pick expert live in the MuJoCo viewer, in real time, looping over seeds.

macOS needs mjpython for the viewer:  mjpython scripts/watch_expert.py [--episodes 20] [--speed 1.0]
"""

import argparse
import time

import mujoco.viewer
import numpy as np

from sim.env import PickEnv
from sim.expert import PickExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = parser.parse_args()

    env = PickEnv(image_size=(48, 64))  # tiny camera renders; the viewer is what you watch
    expert = PickExpert(env)
    rng = np.random.default_rng(args.seed)
    successes = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.2, 0, 0.05]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.8, 135, -25
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            traj, _ = expert.plan(rng)
            for action in traj:
                if not viewer.is_running():
                    return
                start = time.monotonic()
                _, _, _, _, info = env.step(action)
                viewer.sync()
                time.sleep(max(0.0, env.dt / args.speed - (time.monotonic() - start)))
            successes += info["is_success"]
            print(f"episode {ep}: {'success' if info['is_success'] else 'FAIL'}  ({successes}/{ep + 1})")
            time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
