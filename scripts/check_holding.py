"""Check whether each motor holds the arm against gravity, and fingerprint gear ratio by peak speed.

1. Hold test: ramp through rest -> half extend -> extend (offsets from the starting pose, the same
   region exercise_arm.py covers), hold each pose, record load (% of max torque) and sag per joint.
   Aborts home if any load passes ABORT_LOAD, well under the firmware's 80% overload cutoff
   (which drops the motor to 20% torque, i.e. the arm falls).
   --reach adds forward-reach poses that load the shoulder.
2. Speed test: from the half-extend pose, step each joint toward the extend pose and back at full
   servo speed and record peak velocity. Lower gear ratios run faster (1/147 ~2.3x, 1/191 ~1.8x vs 1/345).

Start from the folded rest pose. Keep a hand on the kill switch.

Usage: python scripts/check_holding.py --port /dev/cu.usbmodem... [--reach] [--speed-steps 200 400] [--no-speed]
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MIN_VOLTS = 4.5
RAMP_SPEED = 200  # ticks/s
RATE_HZ = 20
HOLD_S = 3.0
ABORT_LOAD = 60.0  # % of max torque
ABORT_ERR = 200  # ticks

POSES = [
    ("rest", {}),
    ("half extend", {"shoulder_lift": 400, "elbow_flex": -450, "wrist_flex": -200}),
    ("extend", {"shoulder_lift": 800, "elbow_flex": -900, "wrist_flex": -400}),
]
# Forward reach: lift tips the upper arm forward while the elbow straightens, loading the shoulder.
REACH_POSES = [
    ("reach 1", {"shoulder_lift": 1000, "elbow_flex": -1200, "wrist_flex": -400}),
    ("reach 2", {"shoulder_lift": 1200, "elbow_flex": -1500, "wrist_flex": -400}),
    ("reach 3", {"shoulder_lift": 1400, "elbow_flex": -1800, "wrist_flex": -400}),
]
STEP_SIGN = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": -1, "wrist_flex": -1, "wrist_roll": 1, "gripper": 1}


class Abort(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--no-speed", action="store_true", help="skip the speed test")
    parser.add_argument("--reach", action="store_true", help="add forward-reach poses (needs clearance in front)")
    parser.add_argument("--speed-steps", type=int, nargs="+", default=[200], help="step sizes in ticks")
    args = parser.parse_args()

    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(NAMES)}
    bus = FeetechMotorsBus(args.port, motors)
    bus.connect(handshake=False)

    home = bus.sync_read("Present_Position", normalize=False)
    volts = {n: bus.read("Present_Voltage", n, normalize=False) / 10 for n in NAMES}
    print("home:", home)
    if min(volts.values()) < MIN_VOLTS:
        bus.disconnect(disable_torque=False)
        raise SystemExit(f"supply voltage too low: {volts}")

    goal = dict(home)
    peak_load = {n: 0.0 for n in NAMES}

    def sample(check: bool) -> tuple[dict, dict]:
        load = {n: v / 10 for n, v in bus.sync_read("Present_Load", normalize=False).items()}
        pos = bus.sync_read("Present_Position", normalize=False)
        err = {n: pos[n] - goal[n] for n in NAMES}
        for n in NAMES:
            peak_load[n] = max(peak_load[n], abs(load[n]))
        if check:
            worst = max(NAMES, key=lambda n: abs(load[n]))
            if abs(load[worst]) > ABORT_LOAD:
                raise Abort(f"{worst} load {load[worst]:+.1f}%")
            worst = max(NAMES, key=lambda n: abs(err[n]))
            if abs(err[worst]) > ABORT_ERR:
                raise Abort(f"{worst} off by {err[worst]:+d} ticks")
        return load, err

    def ramp_to(target: dict[str, int], check: bool = True) -> None:
        span = max(abs(target[n] - goal[n]) for n in NAMES)
        steps = max(1, int(span / RAMP_SPEED * RATE_HZ))
        start = dict(goal)
        for k in range(1, steps + 1):
            a = k / steps
            goal.update({n: round(start[n] + a * (target[n] - start[n])) for n in NAMES})
            bus.sync_write("Goal_Position", goal, normalize=False)
            sample(check)
            time.sleep(1 / RATE_HZ)

    def pose(offsets: dict[str, int]) -> dict[str, int]:
        return {n: home[n] + offsets.get(n, 0) for n in NAMES}

    bus.sync_write("Goal_Position", goal, normalize=False)
    bus.enable_torque()
    try:
        print(f"\nHOLD TEST (load % of max torque, sag in ticks; abort at {ABORT_LOAD:.0f}%)")
        print(f"{'pose':<12}" + "".join(f"{n:>15}" for n in NAMES))
        for label, offsets in POSES + (REACH_POSES if args.reach else []):
            ramp_to(pose(offsets))
            loads, errs = [], []
            t_end = time.monotonic() + HOLD_S
            while time.monotonic() < t_end:
                load, err = sample(check=True)
                loads.append(load)
                errs.append(err)
                time.sleep(1 / RATE_HZ)
            mean = {n: sum(abs(l[n]) for l in loads) / len(loads) for n in NAMES}
            sag = {n: max((e[n] for e in errs), key=abs) for n in NAMES}
            print(f"{label:<12}" + "".join(f"{mean[n]:>8.1f}% {sag[n]:+5d}" for n in NAMES))

        if not args.no_speed:
            ramp_to(pose(POSES[1][1]))
            print("\nSPEED TEST (peak |velocity|, raw units, step out toward extend and back)")
            print(f"  {'step':<14}" + "".join(f"{st:>8}" for st in args.speed_steps))
            for n in NAMES:
                peaks = []
                for step in args.speed_steps:
                    base = goal[n]
                    peak = 0
                    for target in (base + STEP_SIGN[n] * step, base):
                        goal[n] = target
                        bus.write("Goal_Position", n, target, normalize=False)
                        t_end = time.monotonic() + 0.8
                        while time.monotonic() < t_end:
                            peak = max(peak, abs(bus.read("Present_Velocity", n, normalize=False)))
                    peaks.append(peak)
                    time.sleep(0.3)
                print(f"  {n:<14}" + "".join(f"{pk:>8}" for pk in peaks))
            print("  (peaks that grow with step size are gain-limited, not motor-limited)")
            print("  (a ~1.8-2.3x outlier vs the others suggests a 1/191 or 1/147 motor)")

        ramp_to(home)
    except (Exception, KeyboardInterrupt) as e:  # bus errors too: never drop torque mid-air if we can help it
        print(f"\naborting ({e!r}); returning home")
        ramp_to(home, check=False)
    finally:
        print("\npeak load %:", {n: round(v, 1) for n, v in peak_load.items()})
        print("temps C:", {n: bus.read("Present_Temperature", n, normalize=False) for n in NAMES})
        bus.disable_torque()
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
