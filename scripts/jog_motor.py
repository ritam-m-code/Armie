"""Jog one motor by a small raw-tick offset and return it, to confirm the arm actually moves.

Works on raw ticks (4096/rev), so no calibration is needed. Start with the gripper.

Usage: python scripts/jog_motor.py --port /dev/cu.usbmodem... [--motor gripper] [--delta 100] [--hold 1.0]
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_DELTA = 300  # ~26 deg; anything bigger is not a smoke test
MIN_VOLTS = 4.5  # 7.4 V STS3215 kit runs off a 5 V supply; below this the supply is likely off


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--motor", choices=NAMES, default="gripper")
    parser.add_argument("--delta", type=int, default=100)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--ramp", type=float, default=0.0, help="seconds to ramp to each goal")
    parser.add_argument("--force", action="store_true", help="jog even if supply voltage looks low")
    args = parser.parse_args()
    if abs(args.delta) > MAX_DELTA:
        parser.error(f"--delta must be within +/-{MAX_DELTA}")

    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(NAMES)}
    bus = FeetechMotorsBus(args.port, motors)
    bus.connect(handshake=False)
    name = args.motor
    try:
        volts = bus.read("Present_Voltage", name, normalize=False) / 10
        mode = bus.read("Operating_Mode", name, normalize=False)
        start = bus.read("Present_Position", name, normalize=False)
        print(f"{name}: {volts:.1f} V, mode {mode}, position {start}")
        if mode != 0:
            raise SystemExit(f"{name} is in mode {mode}, not position mode (0); refusing to move")
        if volts < MIN_VOLTS and not args.force:
            raise SystemExit(f"supply {volts:.1f} V < {MIN_VOLTS} V; servo power likely off. Use --force to try anyway")

        target = min(max(start + args.delta, 0), 4095)
        bus.write("Goal_Position", name, start, normalize=False)  # no jump when torque engages
        bus.write("Torque_Enable", name, 1)
        prev = start
        for goal in (target, start):
            steps = max(1, round(args.ramp * 50))  # linear ramp at 50 Hz; 0 = jump straight to goal
            for k in range(1, steps + 1):
                bus.write("Goal_Position", name, round(prev + (goal - prev) * k / steps), normalize=False)
                time.sleep(args.ramp / steps)
            prev = goal
            time.sleep(args.hold)
            now = bus.read("Present_Position", name, normalize=False)
            print(f"goal {goal}  present {now}  error {now - goal:+d}")
    finally:
        bus.write("Torque_Enable", name, 0)
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
