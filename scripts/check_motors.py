"""Read-only bus health check: repeated broadcast pings, then voltage/temperature per motor.

Usage: python scripts/check_motors.py --port /dev/cu.usbmodem... [--scans 3]
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--scans", type=int, default=3)
    args = parser.parse_args()

    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(NAMES)}
    bus = FeetechMotorsBus(args.port, motors)
    bus.connect(handshake=False)
    try:
        expected = set(range(1, len(NAMES) + 1))
        clean = 0
        for scan in range(args.scans):
            found = set(bus.broadcast_ping() or {})
            missing = sorted(expected - found)
            clean += not missing
            print(f"scan {scan + 1}: found {sorted(found)} missing {missing}")
        print(f"clean scans: {clean}/{args.scans}")

        for name in NAMES:
            try:
                volts = bus.read("Present_Voltage", name, normalize=False) / 10
                temp = bus.read("Present_Temperature", name, normalize=False)
                print(f"{motors[name].id} {name:<14} {volts:.1f} V  {temp} C")
            except ConnectionError as e:
                print(f"{motors[name].id} {name:<14} read failed: {e}")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
