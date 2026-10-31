"""Exercise every joint from the current (rest) pose, extend the arm, sweep it, then return home.

Raw ticks (4096/rev), offsets relative to the starting pose, clipped to each motor's firmware
position limits. Moves are linear ramps; any joint lagging its goal by more than MAX_ERR aborts
and the arm ramps back home before torque is released.

Start from the folded rest pose. Keep a hand on the kill switch.

Usage: python scripts/exercise_arm.py --port /dev/cu.usbmodem... [--speed 400]
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MIN_VOLTS = 4.5
MARGIN = 60  # ticks kept away from firmware limits
MAX_ERR = 250  # ticks; bigger lag means stall or collision
RATE_HZ = 50

# (label, offsets from home). Unlisted joints go back to home.
# Rest pose has shoulder_lift at its min and elbow_flex at its max, so +lift / -elbow unfolds.
SEQUENCE = [
    ("pan +", {"shoulder_pan": 300}),
    ("pan -", {"shoulder_pan": -300}),
    ("home", {}),
    ("shoulder lift", {"shoulder_lift": 300}),
    ("home", {}),
    ("elbow flex", {"elbow_flex": -300}),
    ("home", {}),
    ("wrist flex", {"wrist_flex": -300}),
    ("home", {}),
    ("wrist roll +", {"wrist_roll": 400}),
    ("wrist roll -", {"wrist_roll": -400}),
    ("home", {}),
    ("gripper open", {"gripper": 400}),
    ("home", {}),
    ("extend", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400}),
    ("extend, pan +", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400, "shoulder_pan": 400}),
    ("extend, pan -", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400, "shoulder_pan": -400}),
    ("extend, roll", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400, "wrist_roll": 500}),
    ("extend", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400}),
    ("home", {}),
]
EXTEND_START = 14  # index of the first "extend" step


class TrackingError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--speed", type=float, default=400, help="ticks/s for the fastest joint in a move")
    parser.add_argument("--extend-only", action="store_true", help="skip per-joint tests")
    args = parser.parse_args()

    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(NAMES)}
    bus = FeetechMotorsBus(args.port, motors)
    bus.connect(handshake=False)

    home = bus.sync_read("Present_Position", normalize=False)
    lo = {n: bus.read("Min_Position_Limit", n, normalize=False) + MARGIN for n in NAMES}
    hi = {n: bus.read("Max_Position_Limit", n, normalize=False) - MARGIN for n in NAMES}
    lo = {n: min(lo[n], home[n]) for n in NAMES}  # rest pose sits inside the margin; home must stay reachable
    hi = {n: max(hi[n], home[n]) for n in NAMES}
    volts = {n: bus.read("Present_Voltage", n, normalize=False) / 10 for n in NAMES}
    print("home:", home)
    if min(volts.values()) < MIN_VOLTS:
        bus.disconnect(disable_torque=False)
        raise SystemExit(f"supply voltage too low: {volts}")

    goal = dict(home)

    def move_to(target: dict[str, int], check: bool = True) -> None:
        target = {n: min(max(target[n], lo[n]), hi[n]) if n in target else goal[n] for n in NAMES}
        span = max(abs(target[n] - goal[n]) for n in NAMES)
        steps = max(1, int(span / args.speed * RATE_HZ))
        start = dict(goal)
        for k in range(1, steps + 1):
            a = k / steps
            goal.update({n: round(start[n] + a * (target[n] - start[n])) for n in NAMES})
            bus.sync_write("Goal_Position", goal, normalize=False)
            time.sleep(1 / RATE_HZ)
        time.sleep(0.5)
        present = bus.sync_read("Present_Position", normalize=False)
        err = {n: present[n] - goal[n] for n in NAMES}
        print("  err:", " ".join(f"{n}={e:+d}" for n, e in err.items()))
        worst = max(NAMES, key=lambda n: abs(err[n]))
        if check and abs(err[worst]) > MAX_ERR:
            raise TrackingError(f"{worst} off by {err[worst]:+d} ticks")

    bus.sync_write("Goal_Position", goal, normalize=False)  # no jump when torque engages
    bus.enable_torque()
    try:
        sequence = SEQUENCE[EXTEND_START:] if args.extend_only else SEQUENCE
        for label, offsets in sequence:
            print(label)
            move_to({n: home[n] + offsets.get(n, 0) for n in NAMES})
    except (Exception, KeyboardInterrupt) as e:  # bus errors too: never drop torque mid-air if we can help it
        print(f"aborting ({e!r}); returning home")
        move_to(home, check=False)
    finally:
        temps = {n: bus.read("Present_Temperature", n, normalize=False) for n in NAMES}
        print("temps C:", temps)
        bus.disable_torque()
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
