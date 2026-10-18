
import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# =========================================================
# Configuration
# =========================================================
PC_PORT = 5005
PRINT_HZ = 5
RAD_PER_TICK = 2.0 * np.pi / 4096.0

# ESP order -> policy order
ORDER = np.array([1, 0, 2, 3, 4, 6, 5, 7, 8, 9])

# Directions in original ESP order
DIRECTIONS_ESP = np.array(
    [-1, -1, 1, -1, -1, -1, -1, -1, 1, 1],
    dtype=np.float32,
)

JOINT_NAMES = [
    "L hip yaw", "L hip roll", "L hip pitch",
    "L knee", "L ankle",
    "R hip yaw", "R hip roll", "R hip pitch",
    "R knee", "R ankle",
]

HOME = np.array([
    0, -0.087266, 0, -0.523598, 0.523598,
    0, 0.087266, 0, -0.523598, 0.523598,
], dtype=np.float32)

LOG_DIR = Path("sim2real_logs")
LOG_DIR.mkdir(exist_ok=True)

# =========================================================
# Observation processing
# =========================================================

def vec(state, name, size):
    values = state.get(name)
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"Missing/invalid {name}")

    if any(v is None for v in values):
        raise ValueError(f"Null value in {name}")

    result = np.asarray(values, dtype=np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite value in {name}")

    return result


def process_state(state):
    # Positions and velocities in policy order.
    q = vec(state, "q_rad", 10)[ORDER]

    speed_raw = vec(state, "speed_raw", 10)
    dq_servo = (
        speed_raw * DIRECTIONS_ESP * RAD_PER_TICK
    )[ORDER]

    gyro = vec(state, "gyro_rad_s", 3)
    accel = vec(state, "accel", 3)
    quat = vec(state, "quat_wxyz", 4)

    # These are diagnostic observations, not necessarily
    # complete observations from an active PPO rollout.
    obs = np.concatenate([
        gyro,
        accel,
        np.zeros(3, dtype=np.float32),  # command
        np.zeros(2, dtype=np.float32),  # phase
        q - HOME,
        dq_servo,
        np.zeros(10, dtype=np.float32), # previous action
    ]).astype(np.float32)

    assert obs.shape == (41,)

    return q, dq_servo, gyro, accel, quat, obs


def print_dashboard(
    state, q, dq_servo, dq_fd,
    gyro, accel, quat, obs, dt,
):
    print("\033[H\033[J", end="")

    print("MINI PAWO - PASSIVE SENSOR DIAGNOSTICS")
    print("=" * 82)
    print(f"ESP armed:         {state.get('armed')}")
    print(f"Torque enabled:    {state.get('torque_enabled')}")
    print(f"Status:            {state.get('status')}")
    print(f"State sequence:    {state.get('seq')}")
    print(f"Packet interval:   {dt * 1000:.1f} ms"
          if np.isfinite(dt) else "Packet interval: N/A")
    print(f"Sensor sweep:      {state.get('sweep_ms')} ms")
    print(f"IMU age:           {state.get('imu_age_ms')} ms")
    print(f"Joint ages:        {state.get('joint_age_ms')}")

    print("\nIMU")
    print("-" * 82)
    print("Quaternion [w,x,y,z]: ", np.round(quat, 5))
    print("Gyro [rad/s]:        ", np.round(gyro, 5))
    print("Normalized accel:   ", np.round(accel, 5))
    print("Raw accel [m/s^2]:  ",
          state.get("accel_m_s2"))

    print("\nJOINTS")
    print("-" * 82)
    print(
        f"{'Joint':<14} {'q (deg)':>9} "
        f"{'q-HOME':>9} {'Servo dq':>11} "
        f"{'FD dq':>11} {'Diff':>10}"
    )

    for i, name in enumerate(JOINT_NAMES):
        diff = dq_servo[i] - dq_fd[i]
        print(
            f"{name:<14} "
            f"{np.degrees(q[i]):>9.2f} "
            f"{np.degrees(q[i]-HOME[i]):>9.2f} "
            f"{dq_servo[i]:>11.3f} "
            f"{dq_fd[i]:>11.3f} "
            f"{diff:>10.3f}"
        )

    print("\n41-D OBSERVATION (diagnostic)")
    print("-" * 82)

    terms = [
        ("base_ang_vel", 0, 3),
        ("projected_gravity*", 3, 6),
        ("command", 6, 9),
        ("phase*", 9, 11),
        ("joint_pos", 11, 21),
        ("joint_vel", 21, 31),
        ("last_action*", 31, 41),
    ]

    for name, start, end in terms:
        formatted = np.array2string(
            obs[start:end],
            precision=4,
            suppress_small=True,
        )
        print(f"{name:<20}: {formatted}")

    print("\n* Diagnostic placeholder/approximation, not "
          "necessarily the exact mjlab term.")
    print("FD dq = finite-difference joint velocity [rad/s]")
    print("Servo dq = decoded servo speed [rad/s]")
    print("\nNO MOTOR COMMANDS ARE TRANSMITTED. Ctrl+C exits.")


def main():
    filename = (
        LOG_DIR /
        f"passive_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    fields = [
        "host_time", "esp_time_ms", "state_seq",
        "armed", "torque_enabled", "status",
        "sweep_ms", "imu_age_ms", "dt_s",
    ]

    for prefix, count in [
        ("gyro", 3),
        ("accel", 3),
        ("quat", 4),
        ("q", 10),
        ("q_rel", 10),
        ("dq_servo", 10),
        ("dq_fd", 10),
    ]:
        fields += [f"{prefix}_{i}" for i in range(count)]

    fields += [
        f"joint_age_ms_{i}" for i in range(10)
    ]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PC_PORT))
    sock.settimeout(0.5)

    previous_q = None
    previous_esp_ms = None
    previous_session = None
    previous_seq = None
    last_print = 0.0
    count = 0

    print(f"Listening on UDP port {PC_PORT}")
    print(f"Saving to {filename}")
    print("Receive-only mode. No commands will be sent.")

    try:
        with filename.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while True:
                try:
                    packet, address = sock.recvfrom(4096)
                    print(f"Received {len(packet)} bytes from {address}", flush=True)
                except socket.timeout:
                    print("No UDP packet received...", flush=True)
                    continue

                try:
                    state = json.loads(packet)
                except (ValueError, UnicodeDecodeError):
                    continue

                if state.get("type") != "state":
                    continue

                if not state.get("imu_valid", False):
                    continue

                try:
                    q, dq_servo, gyro, accel, quat, obs = (
                        process_state(state)
                    )
                except (ValueError, TypeError):
                    continue

                now = time.monotonic()
                esp_ms = state.get("t_ms")
                session = state.get("session")
                seq = state.get("seq")

                dt = float("nan")
                dq_fd = np.full(10, np.nan, dtype=np.float32)

                # Use ESP timestamps, not variable PC arrival times.
                # Reject duplicate/out-of-order data and reset after reboot.
                if (
                    previous_q is not None
                    and session == previous_session
                    and isinstance(esp_ms, int)
                    and isinstance(previous_esp_ms, int)
                    and isinstance(seq, int)
                    and isinstance(previous_seq, int)
                    and 0 < ((seq - previous_seq) & 0xffffffff) < 0x80000000
                ):
                    delta_ms = (
                        esp_ms - previous_esp_ms
                    ) & 0xffffffff

                    if 0 < delta_ms <= 500:
                        dt = delta_ms / 1000.0
                        dq_fd = (q - previous_q) / dt

                previous_q = q.copy()
                previous_esp_ms = esp_ms
                previous_session = session
                previous_seq = seq

                row = {
                    "host_time": time.time(),
                    "esp_time_ms": esp_ms,
                    "state_seq": seq,
                    "armed": state.get("armed"),
                    "torque_enabled": state.get("torque_enabled"),
                    "status": state.get("status"),
                    "sweep_ms": state.get("sweep_ms"),
                    "imu_age_ms": state.get("imu_age_ms"),
                    "dt_s": dt,
                }

                for prefix, values in [
                    ("gyro", gyro),
                    ("accel", accel),
                    ("quat", quat),
                    ("q", q),
                    ("q_rel", q - HOME),
                    ("dq_servo", dq_servo),
                    ("dq_fd", dq_fd),
                ]:
                    for i, value in enumerate(values):
                        row[f"{prefix}_{i}"] = float(value)

                ages = state.get("joint_age_ms", [])
                for i in range(10):
                    row[f"joint_age_ms_{i}"] = (
                        ages[i] if i < len(ages) else None
                    )

                writer.writerow(row)
                count += 1

                if now - last_print >= 1.0 / PRINT_HZ:
                    print_dashboard(
                        state, q, dq_servo, dq_fd,
                        gyro, accel, quat, obs, dt,
                    )
                    last_print = now
                    f.flush()

    except KeyboardInterrupt:
        print(f"\nStopped. Recorded {count} samples.")
        print(f"Log saved to: {filename}")

    finally:
        sock.close()


if __name__ == "__main__":
    main()
