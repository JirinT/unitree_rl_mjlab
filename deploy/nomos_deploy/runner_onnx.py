import argparse
import json
import socket
import threading
import time

import numpy as np

# ===================== EDIT THESE =====================
ROBOT_IP = "192.168.0.100"
PC_PORT = 5005
ROBOT_PORT = 5006
MODE = "policy"  # "hold", "manual", or "policy"
MODEL_PATH = "logs/rsl_rl/MiniPawo_velocity/50Hz/policy.onnx"
FREQUENCY = 50
TRANSITION_SECONDS = 3
GAIT_PERIOD = 0.4
ACTION_SCALE = 1.0
COMMAND = np.array([0, 0, 0], dtype=np.float32)  # vx, vy, yaw rate

# POLICY order: left yaw, roll, pitch, knee, ankle; right yaw, roll, pitch, knee, ankle.
HOME = np.array([0, -0.087266, 0, -0.523598, 0.523598,
                 0, 0.087266, 0, -0.523598, 0.523598], dtype=np.float32)
MANUAL = np.radians( [-5, 0, 0, -30, 30, 5, 0, 0, -30, 30])
# =====================================================

# Swap yaw/roll on each leg between ESP and policy order. All angles are radians.
ORDER = [1, 0, 2, 3, 4, 6, 5, 7, 8, 9]
DIRECTIONS_ESP = np.array([-1, -1, 1, -1, -1, -1, -1, -1, 1, 1])


class Agent:
    """Replace this class with your agent if its observation/action format differs."""
    def __init__(self, model_path=MODEL_PATH, action_scale=ACTION_SCALE):
        self.action_scale = positive_scale(action_scale)
        import onnxruntime as ort
        self.model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.last_action = np.zeros(10, dtype=np.float32)

    def targets(self, state, elapsed):
        # This is the 41-value observation layout from your colleague's script.
        q = np.array(state["q_rad"], dtype=np.float32)[ORDER]
        # Assumes raw servo speed is ticks/second; change if your hardware uses other units.
        dq = (np.array(state["speed_raw"]) * DIRECTIONS_ESP * (2 * np.pi / 4096))[ORDER]
        phase = 2 * np.pi * elapsed / GAIT_PERIOD
        state["gyro_rad_s"][0] *= -1
        state["accel"][1] *= -1
        obs = np.concatenate([
            state["gyro_rad_s"], state["accel"], COMMAND,
            [0, 0], q - HOME, dq, self.last_action,
        ]).astype(np.float32)
        # obs = np.concatenate([
        #     state["gyro_rad_s"], state["accel"], COMMAND,
        #     [np.sin(phase), np.cos(phase)], q - HOME, dq, self.last_action,
        # ]).astype(np.float32)
        
        # --- Inline Observation Debugger ---
        term_sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "command (twist)": 3,
            "phase": 2,
            "joint_pos": 10,
            "joint_vel": 10,
            "actions": 10,
        }
        
        print("-" * 65)
        print(f"Observation Vector Breakdown (Total Elements: {len(obs)})")
        print("-" * 65)
        
        current_idx = 0
        for term_name, size in term_sizes.items():
            if current_idx >= len(obs):
                break
            term_data = obs[current_idx : current_idx + size]
            formatted_data = ", ".join([f"{x:>11.6f}" for x in term_data])
            print(f"{term_name:<20} | [{formatted_data}]")
            current_idx += size
            
        remaining_elements = len(obs) - current_idx
        if remaining_elements > 0:
            print("-" * 65)
            print(f"{'height_scan/other':<20} | {remaining_elements} elements")
            preview = ", ".join([f"{x:>11.6f}" for x in obs[current_idx:current_idx+5]])
            print(f"{'':<20} | [{preview}, ...]")
        print("-" * 65)
        # -----------------------------------
        
        action = self.model.run([self.output_name], {self.input_name: obs[None]})[0][0]
        self.last_action = action.copy()
        # Keep raw action history, as in training; scale only the joint offset.
        return HOME + action * self.action_scale


def positive_scale(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("action scale must be a finite positive number") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("action scale must be a finite positive number")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniPawo robot ONNX runner")
    parser.add_argument("--onnx", default=MODEL_PATH, help="ONNX policy path")
    parser.add_argument("--action-scale", type=positive_scale, default=ACTION_SCALE,
                        help="Joint offset scale; must match training (default: %(default)s)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    # Manual/hold modes do not load the model.
    agent = Agent(args.onnx, args.action_scale) if MODE == "policy" else None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PC_PORT))
    sock.settimeout(0.1)
    latest = None
    running = True

    # 1. Create a thread event to signal when fresh data arrives
    new_data_event = threading.Event()

    def receive():
        nonlocal latest
        while running:
            try:
                data, _ = sock.recvfrom(2048)
                latest = json.loads(data)  # Replace old state; never queue inference jobs.
                
                # 2. Unblock the main thread immediately
                new_data_event.set()
                
            except socket.timeout:
                pass
            except (ValueError, OSError):
                pass

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    print(f"Mode: {MODE}. Listening on {PC_PORT}, sending to {ROBOT_IP}:{ROBOT_PORT}.")
    if agent is not None:
        print(f"Policy: {args.onnx} | action scale: {agent.action_scale:g} | targets = HOME + scale * raw action")
    print("Type arm in ESP Serial Monitor. Ctrl+C exits.")
    
    # [Removed fixed frequency deadline variables]
    start_pose = None
    started = None
    phase_started = None
    key = None
    seq = 0
    previous_status = None
    
    try:
        while True:
            # 3. Block execution until the receiver thread flags new data
            if not new_data_event.wait(timeout=0.1):
                continue  # Timeout allows KeyboardInterrupts to register if hardware disconnects
            
            new_data_event.clear()  # Reset flag for the next packet
            
            state = latest  # Guaranteed to be a fresh snapshot
            now = time.monotonic()
            
            if state is not None:
                status = (state["armed"], state["status"])
                if status != previous_status:
                    print("ESP:", status)
                    previous_status = status
                current_key = (state["session"], state["epoch"])
                if current_key != key:
                    key = current_key
                    start_pose = started = phase_started = None
                    seq = state.get("applied_seq") or 0
                    if agent is not None:
                        agent.last_action.fill(0)
                if state["armed"]:
                    if start_pose is None:
                        start_pose = np.array(state["q_rad"], dtype=np.float32)[ORDER]
                        started = now
                    elapsed = now - started
                    if MODE == "hold":
                        targets = start_pose
                    elif elapsed < TRANSITION_SECONDS:
                        goal = MANUAL if MODE == "manual" else HOME
                        alpha = elapsed / TRANSITION_SECONDS
                        targets = start_pose + alpha * (goal - start_pose)
                    elif MODE == "manual":
                        targets = MANUAL
                    else:
                        if phase_started is None:
                            phase_started = now
                        targets = agent.targets(state, now - phase_started)
                    applied = state.get("applied_seq")
                    if applied is not None and 0 < ((applied - seq) & 0xffffffff) < 0x80000000:
                        seq = applied
                    seq = (seq + 1) & 0xffffffff
                    sock.sendto(json.dumps({
                        "cmd": "targets",
                        "session": state["session"], "epoch": state["epoch"],
                        "seq": seq, "state_seq": state["seq"],
                        "targets_rad": np.asarray(targets)[ORDER].tolist(),
                    }).encode(), (ROBOT_IP, ROBOT_PORT))
                else:
                    start_pose = started = phase_started = None
                    
            # 4. [Removed time.sleep(remaining) and deadline calculations]
            # The loop will naturally block at new_data_event.wait() at the top of the next iteration.
            
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        # A single stop request on exit; the ESP watchdog handles missing commands too.
        try:
            if latest is not None:
                sock.sendto(json.dumps({"cmd": "stop", "session": latest["session"]}).encode(),
                            (ROBOT_IP, ROBOT_PORT))
        finally:
            running = False
            receiver.join(timeout=0.2)
            sock.close()

if __name__ == "__main__":
    main()