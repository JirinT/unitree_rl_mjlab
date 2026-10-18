"""Receive-only MiniPawo ONNX probe. This module never transmits robot commands.

Requires numpy, onnxruntime and matplotlib. Command and phase are zero.
Accel is a gravity proxy, not quaternion-derived projected gravity.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "nomos_deploy"))
from imu_observations import quaternion_imu, unit_quaternion

ORDER = np.array([1, 0, 2, 3, 4, 6, 5, 7, 8, 9])
DIRECTIONS = np.array([-1, -1, 1, -1, -1, -1, -1, -1, 1, 1], np.float32)
HOME = np.array([0, -.087266, 0, -.523598, .523598,
                 0, .087266, 0, -.523598, .523598], np.float32)
JOINTS = ["L yaw", "L roll", "L pitch", "L knee", "L ankle",
          "R yaw", "R roll", "R pitch", "R knee", "R ankle"]


def vector(state, key, size):
    value = state.get(key)
    if not isinstance(value, list) or len(value) != size or any(x is None for x in value):
        missing = [i for i, x in enumerate(value) if x is None] if isinstance(value, list) else []
        raise ValueError(f"Invalid {key}; expected {size} values; null indices={missing}")
    a = np.array(value, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {key}")
    return a


def observation(state, convention, gravity_source="accel", quaternion_direction="sensor-to-world", imu_to_body=(1.,0.,0.,0.)):
    q = vector(state, "q_rad", 10)[ORDER]
    dq = (vector(state, "speed_raw", 10) * DIRECTIONS * (2*np.pi/4096))[ORDER]
    if gravity_source == "quaternion":
        gyro, gravity = quaternion_imu(state, quaternion_direction, imu_to_body)
    else:
        gyro = vector(state, "gyro_rad_s", 3)
        gravity = vector(state, "accel", 3)
        if convention == "deployment":
            gyro[0] *= -1
            gravity[1] *= -1
    # ONNX includes normalization. Do not normalize these inputs a second time.
    obs = np.concatenate([gyro, gravity, np.zeros(5, np.float32), q-HOME, dq,
                          np.zeros(10, np.float32)]).astype(np.float32)
    return obs, q, dq


class Predictor:
    def __init__(self, session):
        ins, outs = session.get_inputs(), session.get_outputs()
        if len(ins) != 1 or len(outs) != 1 or ins[0].shape != [1, 41] or outs[0].shape != [1, 10]:
            raise ValueError("Expected a single-input/output ONNX model [1,41] -> [1,10]")
        self.session, self.input, self.output = session, ins[0].name, outs[0].name

    def __call__(self, obs):
        action = np.asarray(self.session.run([self.output], {self.input: obs[None]})[0])
        if action.shape != (1, 10) or not np.isfinite(action).all():
            raise ValueError("Policy returned invalid/nonfinite actions")
        return action[0].copy()


def receive_packets(bind, port, deadline):
    # Deliberately no connect, send, sendto, arming, stop or torque operations.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((bind, port))
        sock.settimeout(.25)
        while time.monotonic() < deadline:
            try:
                packet, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                state = json.loads(packet)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(state, dict):
                yield state
    finally:
        sock.close()


def record(packets, predict, csv_path, convention="deployment", scale=1., max_samples=0, gravity_source="accel", quaternion_direction="sensor-to-world", imu_to_body=(1.,0.,0.,0.)):
    fields = ["elapsed_s", "host_time", "esp_time_ms", "session", "epoch", "state_seq",
              "armed", "torque_enabled", "status", "imu_age_ms", "sweep_ms",
              "feedback_reset", "gravity_proxy_norm", "max_abs_joint_velocity"]
    groups = {"q":10, "dq":10, "obs_zero":41, "obs_feedback":41,
              "action_zero":10, "action_feedback":10, "target_zero":10, "target_feedback":10}
    for stem, size in groups.items():
        fields += [f"{stem}_{i}" for i in range(size)]
    previous = np.zeros(10, np.float32)
    key = last_seq = None
    start = time.monotonic()
    accepted = rejected = 0
    reasons = Counter()
    examples = []
    def reject(reason, state):
        nonlocal rejected
        rejected += 1
        reasons[reason] += 1
        if reasons[reason] == 1:
            print(f"[SKIP] {reason}; motor IDs={state.get('ids')}", flush=True)
        if len(examples) < 5:
            examples.append(dict(reason=reason, state=state))
    with Path(csv_path).open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        try:
            for state in packets:
                if state.get("type") != "state":
                    reject(f"Unexpected packet type: {state.get('type')!r}", state)
                    continue
                if state.get("imu_valid") is False:
                    reject("imu_valid=false", state)
                    continue
                seq = state.get("seq")
                if not isinstance(seq, int) or isinstance(seq, bool):
                    reject("Missing/invalid seq", state)
                    continue
                state_key = (state.get("session"), state.get("epoch"))
                same = state_key == key and last_seq is not None
                delta = ((seq-last_seq) & 0xffffffff) if same else None
                if same and (delta == 0 or delta >= 0x80000000):
                    reject("Duplicate/out-of-order seq", state)
                    continue
                try:
                    obs, q, dq = observation(state, convention, gravity_source, quaternion_direction, imu_to_body)
                except (ValueError, TypeError, OverflowError) as exc:
                    reject(str(exc), state)
                    continue
                reset = not same or delta != 1
                if reset:
                    previous.fill(0)
                feedback_obs = obs.copy()
                feedback_obs[31:41] = previous
                action_zero = predict(obs)
                action_feedback = predict(feedback_obs)
                row = dict(elapsed_s=time.monotonic()-start, host_time=time.time(),
                           esp_time_ms=state.get("t_ms", ""), session=state.get("session", ""),
                           epoch=state.get("epoch", ""), state_seq=seq,
                           armed=state.get("armed", ""), torque_enabled=state.get("torque_enabled", ""),
                           status=state.get("status", ""), imu_age_ms=state.get("imu_age_ms", ""),
                           sweep_ms=state.get("sweep_ms", ""), feedback_reset=int(reset),
                           gravity_proxy_norm=float(np.linalg.norm(obs[3:6])),
                           max_abs_joint_velocity=float(np.abs(dq).max()))
                arrays = dict(q=q, dq=dq, obs_zero=obs, obs_feedback=feedback_obs,
                              action_zero=action_zero, action_feedback=action_feedback,
                              target_zero=HOME+scale*action_zero, target_feedback=HOME+scale*action_feedback)
                for stem, values in arrays.items():
                    row.update({f"{stem}_{i}": float(v) for i, v in enumerate(values)})
                writer.writerow(row)
                file.flush()
                previous = action_feedback.copy()
                key, last_seq = state_key, seq
                accepted += 1
                if accepted % 50 == 1:
                    print(f"samples={accepted} skipped={rejected} "
                          f"max|action| zero={np.abs(action_zero).max():.3f} "
                          f"feedback={np.abs(action_feedback).max():.3f}", flush=True)
                if max_samples and accepted >= max_samples:
                    break
        except KeyboardInterrupt:
            print("Stopped; preserving captured samples.")
    report = dict(accepted=accepted, rejected=rejected, reasons=dict(reasons), examples=examples)
    Path(csv_path).with_name("rejections.json").write_text(json.dumps(report, indent=2))
    if reasons:
        print("Rejection counts:", dict(reasons))
    return accepted, rejected


def plot(csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No valid state packets recorded. Check the sender, port, and packet schema.")
        return
    t = [float(r["elapsed_s"]) for r in rows]
    gravity = np.array([[float(r[f"obs_zero_{i}"]) for i in (3, 4, 5)] for r in rows])
    metadata_path = Path(csv_path).with_name("metadata.json")
    source = "unknown"
    if metadata_path.exists():
        source = json.loads(metadata_path.read_text()).get("gravity_source", "accel")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for i, axis in enumerate("XYZ"):
        axes[0].plot(t, gravity[:, i], label=axis, lw=1.)
    axes[0].set_ylabel("Gravity observation (unitless)")
    axes[0].legend()
    axes[1].plot(t, np.linalg.norm(gravity, axis=1), label="Vector magnitude")
    axes[1].axhline(1., color="black", ls="--", lw=.8, label="Unit length")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Host elapsed time (s)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(f"RECEIVE ONLY — Gravity supplied to policy (source: {source})")
    fig.savefig(Path(csv_path).with_name("gravity.png"), dpi=150)
    plt.close(fig)
    for kind, ylabel in [("action", "Raw action"), ("target", "Hypothetical target (rad)")]:
        fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True, layout="constrained")
        for i, name in enumerate(JOINTS):
            ax = axes[i % 5, i // 5]
            for mode, label in [("zero", "Previous action = zero"), ("feedback", "Predicted-action feedback")]:
                ax.plot(t, [float(r[f"{kind}_{mode}_{i}"]) for r in rows], label=label, lw=.9)
            if kind == "target":
                ax.plot(t, [float(r[f"q_{i}"]) for r in rows], label="Measured position", color="black", lw=.7)
            ax.set_title(name); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7)
        for ax in axes[-1]: ax.set_xlabel("Host elapsed time (s)")
        fig.suptitle("RECEIVE ONLY — " + ("Policy outputs" if kind == "action" else "Computed targets, never transmitted"))
        fig.savefig(Path(csv_path).with_name(kind+"s.png"), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--duration", type=float, default=30.)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--imu-convention", choices=["deployment", "passive"], default="deployment",
                   help="deployment reproduces runner_onnx gyro-X/accel-Y flips; passive applies no flips")
    p.add_argument("--gravity-source", choices=["accel", "quaternion"], default="accel")
    p.add_argument("--quaternion-direction", choices=["sensor-to-world", "world-to-sensor"], default="sensor-to-world")
    p.add_argument("--imu-to-body-quat", type=float, nargs=4, default=(1.,0.,0.,0.), metavar=("W","X","Y","Z"))
    p.add_argument("--action-scale", type=float, default=1.)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    if not np.isfinite(a.duration) or a.duration <= 0 or not np.isfinite(a.action_scale) or a.action_scale <= 0 or a.max_samples < 0:
        p.error("duration/scale must be finite positive values; max-samples must be nonnegative")
    try:
        unit_quaternion(a.imu_to_body_quat)
    except ValueError as exc:
        p.error(str(exc))
    import onnxruntime as ort
    predict = Predictor(ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"]))
    out = a.out or Path("sim2real_logs") / datetime.now().strftime("passive_policy_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(onnx=str(a.onnx.resolve()), sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
                    imu_convention=a.imu_convention, action_scale=a.action_scale,
                    gravity_source=a.gravity_source, quaternion_direction=a.quaternion_direction,
                    imu_to_body_quat=list(a.imu_to_body_quat),
                    joint_names=JOINTS, order=ORDER.tolist(), home=HOME.tolist(),
                    semantics="No robot commands. Zero command/phase. Both previous-action lanes start at zero. "
                              "Feedback lane resets on session/epoch changes or sequence gaps. "
                              "Predicted feedback is hypothetical, not physical closed-loop control. "
                              "Accel is a gravity proxy; speed_raw is assumed signed ticks/s before direction conversion. "
                              "Packet stream is sampled as received; no fixed-rate resampling or static-state assertion.")
    if a.gravity_source == "quaternion":
        metadata["semantics"] = metadata["semantics"].replace("Accel is a gravity proxy;", "Gravity is world-down inverse-rotated through quat_wxyz then sensor-to-body mounting; legacy IMU sign flips are NOT applied;")
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"RECEIVE ONLY: UDP {a.bind}:{a.port}, {a.duration:g}s. Output: {out}")
    print("Stop other listeners on this port. Keep command-sending runners stopped; this script cannot disable existing torque.")
    packets = receive_packets(a.bind, a.port, time.monotonic()+a.duration)
    try:
        accepted, rejected = record(packets, predict, out/"observations_actions.csv", a.imu_convention, a.action_scale, a.max_samples, a.gravity_source, a.quaternion_direction, a.imu_to_body_quat)
    finally:
        packets.close()
    plot(out/"observations_actions.csv")
    print(f"Saved {accepted} samples; skipped {rejected}. Output: {out.resolve()}")

if __name__ == "__main__":
    main()
