"""Read-only: average joint positions in LeRobot's calibrated units (body joints in degrees, gripper 0-100).

Never writes to the motors, so torque stays as it is (off at rest). Pose the arm by hand, then run.

Usage: python scripts/read_joints.py --port /dev/cu.usbmodem... [--id follower] [--samples 30]
       python scripts/read_joints.py --port ... --log 30 [--out outputs/joint_log.csv]   # record while moving by hand
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="follower")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--log", type=float, default=0, help="record for this many seconds at 10 Hz instead")
    parser.add_argument("--out", type=Path, default=Path("outputs/joint_log.csv"))
    args = parser.parse_args()

    calib = {k: MotorCalibration(**v) for k, v in json.loads((CALIB_DIR / f"{args.id}.json").read_text()).items()}
    motors = {
        name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)
        for i, name in enumerate(NAMES)
    }
    bus = FeetechMotorsBus(args.port, motors, calibration=calib)
    bus.connect(handshake=False)
    try:
        torque = bus.sync_read("Torque_Enable", normalize=False)
        reads, stamps = [], []
        n, period = (round(args.log * 10), 0.1) if args.log else (args.samples, 0.02)
        # Stopping early (Ctrl+C or SIGTERM) keeps what was recorded so far.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        t0 = time.monotonic()
        try:
            for _ in range(n):
                pos = bus.sync_read("Present_Position")
                reads.append([pos[name] for name in NAMES])
                stamps.append(time.monotonic() - t0)
                time.sleep(period)
        except KeyboardInterrupt:
            print(f"stopped early after {len(reads)} samples")
    finally:
        bus.disconnect(disable_torque=False)

    reads = np.array(reads)
    if args.log:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.out, np.column_stack([stamps, reads]), delimiter=",", header="t," + ",".join(NAMES), comments="", fmt="%.3f")
        print(args.out)
        print(f"{'joint':<14}{'start':>9}{'min':>9}{'max':>9}{'end':>9}")
        for i, name in enumerate(NAMES):
            col = reads[:, i]
            print(f"{name:<14}{col[0]:>9.1f}{col.min():>9.1f}{col.max():>9.1f}{col[-1]:>9.1f}")
        return
    mean, spread = reads.mean(axis=0), reads.max(axis=0) - reads.min(axis=0)
    print(f"{'joint':<14}{'mean':>9}{'spread':>9}  torque")
    for i, name in enumerate(NAMES):
        unit = "" if name == "gripper" else " deg"
        print(f"{name:<14}{mean[i]:>9.2f}{spread[i]:>9.2f}{unit}  {torque[name]}")


if __name__ == "__main__":
    main()
