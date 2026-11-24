"""Record the scripted pick expert to an MP4 from a free orbit camera (offscreen, no viewer needed).

Usage: python scripts/record_expert.py [--episodes 5] [--out outputs/expert.mp4] [--size 720 1280]
"""

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from sim.env import PickEnv
from sim.expert import PickExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/expert.mp4")
    parser.add_argument("--size", type=int, nargs=2, default=[720, 1280], metavar=("H", "W"))
    args = parser.parse_args()

    env = PickEnv(image_size=(48, 64))
    expert = PickExpert(env)
    rng = np.random.default_rng(args.seed)
    h, w = args.size
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, w)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(env.model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 135, -25

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = round(1 / env.dt)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )

    def frame() -> None:
        renderer.update_scene(env.data, cam)
        ffmpeg.stdin.write(renderer.render().tobytes())

    successes = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        traj, _ = expert.plan(rng)
        for _ in range(fps // 2):  # half-second still at episode start
            frame()
        for action in traj:
            _, _, _, _, info = env.step(action)
            frame()
        for _ in range(fps // 2):
            frame()
        successes += info["is_success"]
        print(f"episode {ep}: {'success' if info['is_success'] else 'FAIL'}")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()
    env.close()
    print(f"{successes}/{args.episodes} success -> {out}")


if __name__ == "__main__":
    main()
