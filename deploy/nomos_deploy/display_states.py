#!/usr/bin/env python3
import json
import math
import socket
import time

# EDIT if needed. No command-line arguments, no extra packages.
ROBOT_IP = "192.168.0.100"
PC_PORT = 5005
REFRESH_SECONDS = 0.1

# Firmware order: ID, name, encoder-to-simulation direction.
MOTORS = [
    (2, "left_hip_roll", -1), (3, "left_hip_yaw", -1),
    (4, "left_hip_pitch", 1), (5, "left_knee", -1), (6, "left_ankle", -1),
    (11, "right_hip_roll", -1), (12, "right_hip_yaw", -1),
    (13, "right_hip_pitch", -1), (14, "right_knee", 1), (15, "right_ankle", 1),
]
# Assumes speed_raw is signed encoder ticks/second.
# This converts it to simulation-coordinate rad/s, including direction.
RAD_PER_SPEED_UNIT = 2 * math.pi / 4096


def number(value, width=10):
    if value is None:
        return "--".rjust(width)
    return f"{value:{width}.4f}"


def vector(state, key, units):
    values = state.get(key)
    text = "--" if values is None else "[" + ", ".join(number(v).strip() for v in values) + "]"
    return f"{key:16s} {text:42s} {units}"


def render(state):
    lines = [
        "ROBOT STATE — DISPLAY ONLY, NO MOTOR COMMANDS",
        f"session={state.get('session')} epoch={state.get('epoch')} seq={state.get('seq')} "
        f"armed={state.get('armed')} torque={state.get('torque_enabled')}",
        f"status={state.get('status')} applied_seq={state.get('applied_seq')} sweep={state.get('sweep_ms')} ms",
        "",
        " ID  JOINT                    ANGLE rad   VELOCITY rad/s  RAW SPEED  AGE ms",
        "-------------------------------------------------------------------------",
    ]
    ids = state.get("ids", [])
    angles = state.get("q_rad", [])
    speeds = state.get("speed_raw", [])
    ages = state.get("joint_age_ms", [])
    for motor_id, name, direction in MOTORS:
        index = ids.index(motor_id) if motor_id in ids else -1
        q = angles[index] if 0 <= index < len(angles) else None
        raw = speeds[index] if 0 <= index < len(speeds) else None
        age = ages[index] if 0 <= index < len(ages) else None
        velocity = None if raw is None else raw * direction * RAD_PER_SPEED_UNIT
        lines.append(f"{motor_id:3d}  {name:21s} {number(q)}  {number(velocity,14)}  "
                     f"{str(raw) if raw is not None else '--':>9s}  {str(age) if age is not None else '--':>6s}")
    lines.extend([
        "",
        f"IMU valid={state.get('imu_valid')} age={state.get('imu_age_ms', '--')} ms",
        vector(state, "quat_wxyz", "unitless; w,x,y,z"),
        vector(state, "gyro_rad_s", "rad/s; sensor x,y,z"),
        vector(state, "accel_m_s2", "m/s²; sensor x,y,z"),
        vector(state, "accel", "unit vector; sensor x,y,z"),
        "",
        "Joint angle/velocity use simulation signs. IMU stays in the sensor frame.",
        "Velocity assumes raw units=ticks/s. Stop other UDP clients using port 5005.",
        "No targets or stop packets are sent. Ctrl+C exits.",
    ])
    return "\n".join(lines)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PC_PORT))
    sock.settimeout(1)
    last_print = 0
    last_received = 0
    latest = None
    print(f"Listening for {ROBOT_IP}:5006 on UDP {PC_PORT}...")
    try:
        while True:
            try:
                data, peer = sock.recvfrom(2048)
            except socket.timeout:
                age = "never received" if latest is None else f"last packet {time.monotonic()-last_received:.1f}s ago"
                print(f"\nNO TELEMETRY ({age}). Check ROBOT_IP, firmware PC_IP and Wi-Fi.", flush=True)
                continue
            if peer != (ROBOT_IP, 5006):
                continue
            try:
                state = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            if state.get("type") != "state":
                continue
            latest = state
            last_received = time.monotonic()
            if last_received-last_print >= REFRESH_SECONDS:
                # ANSI redraw; receive all intervening packets rather than sleeping.
                print("\033[2J\033[H" + render(latest), flush=True)
                last_print = time.monotonic()
    except KeyboardInterrupt:
        print("\nDisplay closed.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()