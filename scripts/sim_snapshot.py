"""Render the front and wrist cameras of the sim scene to PNGs.

Usage: python scripts/sim_snapshot.py [--qpos q1 ... q6] [--out outputs/snapshots]
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).resolve().parents[1] / "sim" / "assets" / "scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", type=float, nargs=6, default=[0.0] * 6, help="arm joints, rad")
    parser.add_argument("--out", type=Path, default=Path("outputs/snapshots"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:6] = args.qpos
    mujoco.mj_forward(model, data)

    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, 480, 640) as renderer:
        for cam in ("front", "wrist"):
            renderer.update_scene(data, camera=cam)
            path = args.out / f"{cam}.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)


if __name__ == "__main__":
    main()
