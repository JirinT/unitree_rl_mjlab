"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_cg = None


def _load_keyboard():
  global _cg
  if _cg is not None:
    return
  if sys.platform != "darwin":
    raise ValueError("--keyboard-control uses macOS CoreGraphics; omit it on other platforms")
  import ctypes
  import ctypes.util
  library = ctypes.util.find_library("CoreGraphics") or (
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
  )
  _cg = ctypes.CDLL(library)
  _cg.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
  _cg.CGEventSourceKeyState.restype = ctypes.c_bool


def _down(vk: int) -> bool:
  return bool(_cg.CGEventSourceKeyState(1, vk))  # 1 = HID system state


# macOS virtual keycodes (physical key positions)
VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT = 0x7E, 0x7D, 0x7B, 0x7C
VK_COMMA, VK_PERIOD = 0x2B, 0x2F
VK_SHIFT, VK_OPTION = 0x38, 0x3A


class KeyboardTwist:
  """Hold UP/DOWN: forward/back. LEFT/RIGHT: yaw. , / . : strafe.
  Speed: normal by default, hold SHIFT for fast, OPTION for slow."""

  def __init__(self, max_fwd=0.6, max_back=0.5, max_side=0.5, max_yaw=2,
               slow=0.3, normal=0.4, fast=1.0):
    _load_keyboard()
    self.max_fwd, self.max_back = max_fwd, max_back
    self.max_side, self.max_yaw = max_side, max_yaw
    self.slow, self.normal, self.fast = slow, normal, fast
    self._last_printed = None

  def apply(self, env) -> None:  # main thread, once per step
    s = self.fast if _down(VK_SHIFT) else self.slow if _down(VK_OPTION) else self.normal
    vx = s * (self.max_fwd * _down(VK_UP) - self.max_back * _down(VK_DOWN))
    vy = s * self.max_side * (_down(VK_COMMA) - _down(VK_PERIOD))
    wz = s * self.max_yaw * (_down(VK_LEFT) - _down(VK_RIGHT))

    if (vx, vy, wz) != self._last_printed:
      self._last_printed = (vx, vy, wz)
      print(f"cmd vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}")

    term = env.unwrapped.command_manager.get_term("twist")
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = vy
    term.vel_command_b[:, 2] = wz
    term.time_left[:] = 1e9
    for flag in ("is_standing_env", "is_heading_env"):
      if hasattr(term, flag):
        getattr(term, flag)[:] = False


class CommandedPolicy:
  """Optional keyboard commands without a robot-specific observation layout."""

  def __init__(self, policy, env, keyboard):
    from scripts.keyboard_hand_control import CommandObservationRefresh
    self.policy, self.env, self.keyboard = policy, env, keyboard
    self.refresh_commands = CommandObservationRefresh(env)

  def __call__(self, obs):
    self.keyboard.apply(self.env)
    return self.policy(self.refresh_commands(obs))

  def reset(self, dones=None):
    if hasattr(self.policy, "reset"):
      self.policy.reset(dones)

  def close(self):
    try:
      if hasattr(self.keyboard, "close"):
        self.keyboard.close()
    finally:
      if hasattr(self.policy, "close"):
        self.policy.close()


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  onnx_file: str | None = None
  """Run a feedforward actor ONNX directly; its dimensions must match the task."""
  action_scale: float | None = None
  """Override joint action scale; must match policy training."""
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "native"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  keyboard_control: bool = False
  """Hand tracking / arm intervention: terminal walking and arm controls on macOS/Linux. Other tasks: legacy macOS hold-to-walk controls."""
  arm_control: Literal["random", "policy", "demo"] | None = None
  """ArmIntervention only: sampled overrides, PPO-only arms, or a repeating deterministic 32-second demo. Omit with --keyboard-control."""

  diagnostic_command: tuple[float, float, float] | None = None
  """Pin vx, vy, yaw-rate for repeatable simulation diagnostics."""
  policy_decimation: int | None = None
  """Override physics steps per policy step; use the training value."""
  actuator_model: Literal["config", "builtin", "bam-sts3215", "bam-waveshare-st3215"] = "config"
  """Preserve the registered recipe by default; overrides are MiniPawo-only."""
  bam_voltage: float = 7.4
  bam_kp_fw: float | None = None
  """Firmware P gain override for explicit BAM CLI selection."""
  bam_command_delay_min: int = 3  # Physics steps; 15 ms at dt=0.005
  bam_command_delay_max: int = 6  # Physics steps; 30 ms at dt=0.005
  bam_friction_model: Literal["m1", "m2", "m3", "m4", "m5", "m6"] = "m6"
  seed: int | None = None
  sim_log: str | None = None
  """Optional detailed MiniPawo CSV diagnostics; omit for other robots."""

  plot_actions: bool = False
  """Save MiniPawo env-0 raw actions and joint-target PNGs when playback exits; enables --sim-log automatically."""

  plot_observations: bool = False
  """Save all 41 MiniPawo actor-input plots on exit; enables --sim-log automatically."""

  compare_onnx: str | None = None
  """Compare raw deterministic PT actions against this ONNX model."""
  comparison_csv: str = "sim2real_logs/pt_onnx_comparison.csv"
  headless_steps: int = 0
  """Run a finite simulation without a viewer (requires a positive step count)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def prepare_env_cfg(task_id: str, cfg: PlayConfig):
  """Load the selected playback recipe and apply only explicit CLI overrides."""
  from copy import deepcopy
  import math

  env_cfg = deepcopy(load_env_cfg(task_id, play=True))
  from src.tasks.velocity.mdp.arm_intervention import ArmInterventionActionCfg
  arm_intervention = isinstance(env_cfg.actions.get("joint_pos"), ArmInterventionActionCfg)
  if cfg.arm_control is not None:
    if not arm_intervention:
      raise ValueError("--arm-control requires an ArmIntervention task")
    if cfg.keyboard_control:
      raise ValueError("Use --keyboard-control alone for manual arm takeover, or --arm-control without keyboard")
    if cfg.arm_control == "demo" and cfg.diagnostic_command is not None:
      raise ValueError("The arm demo supplies walking commands; omit --diagnostic-command")
    if cfg.arm_control == "demo":
      from scripts.arm_intervention_control import configure_demo_arm_actuators
      configure_demo_arm_actuators(env_cfg.scene.entities["robot"])
  if arm_intervention and (cfg.keyboard_control or cfg.arm_control in {"policy", "demo"}):
    env_cfg.actions["joint_pos"].mode_probabilities = (1., 0., 0., 0.)
  mini_pawo = task_id == "MiniPawo-Flat" or task_id.startswith("MiniPawo-Flat-")
  if (cfg.sim_log or cfg.compare_onnx or cfg.plot_actions or cfg.plot_observations) and not mini_pawo:
    raise ValueError("--sim-log, --plot-actions, --plot-observations and --compare-onnx are MiniPawo diagnostics")
  if cfg.headless_steps < 0:
    raise ValueError("headless_steps must be nonnegative")
  if cfg.keyboard_control and cfg.diagnostic_command is not None:
    raise ValueError("Use either --diagnostic-command or --keyboard-control")
  if (cfg.keyboard_control or cfg.diagnostic_command is not None) and "twist" not in env_cfg.commands:
    raise ValueError("Velocity command overrides require a task with a twist command")
  if cfg.action_scale is not None:
    if not math.isfinite(cfg.action_scale) or cfg.action_scale <= 0:
      raise ValueError("action_scale must be finite and positive")
    if "joint_pos" not in env_cfg.actions:
      raise ValueError("--action-scale requires a joint_pos action term")
    env_cfg.actions["joint_pos"].scale = cfg.action_scale
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed
  if cfg.policy_decimation is not None:
    if cfg.policy_decimation < 1:
      raise ValueError("policy_decimation must be positive")
    env_cfg.decimation = cfg.policy_decimation
  if cfg.num_envs is not None:
    if cfg.num_envs < 1:
      raise ValueError("num_envs must be positive")
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  if cfg.camera is not None:
    raise ValueError("--camera is not supported by this mjlab viewer; configure viewer lookat/body instead")
  if cfg.no_terminations:
    env_cfg.terminations = {}
  if cfg.diagnostic_command is not None:
    if not all(math.isfinite(x) for x in cfg.diagnostic_command):
      raise ValueError("diagnostic command must be finite")
    cmd = env_cfg.commands["twist"]
    vx, vy, wz = cfg.diagnostic_command
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (wz, wz)
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_standing_envs = float(vx == vy == wz == 0)

  actuator_info = {}
  if cfg.actuator_model != "config":
    if not mini_pawo:
      raise ValueError("--actuator-model overrides are only supported for MiniPawo tasks")
    from src.assets.robots.mini_pawo.actuator_cfg import configure_actuators
    # Legacy helper expects this base task ID; apply to the selected recipe cfg.
    actuator_info = configure_actuators(
      "MiniPawo-Flat", env_cfg, cfg.actuator_model, cfg.bam_voltage,
      cfg.bam_friction_model, cfg.bam_command_delay_min,
      cfg.bam_command_delay_max, kp_fw=cfg.bam_kp_fw,
    )
  elif mini_pawo:
    from src.assets.robots.mini_pawo.actuator_cfg import describe_actuators
    actuator_info = describe_actuators(env_cfg.scene.entities["robot"])
  return env_cfg, actuator_info


def run_play(task_id: str, cfg: PlayConfig):
  if (cfg.plot_actions or cfg.plot_observations) and cfg.sim_log is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cfg = replace(cfg, sim_log=str(Path("sim2real_logs") / f"sim_play_{stamp}.csv"))
  configure_torch_backends()
  onnx_mode = cfg.onnx_file is not None
  dummy_mode = cfg.agent in {"zero", "random"}
  if onnx_mode and (dummy_mode or cfg.checkpoint_file or cfg.compare_onnx):
    raise ValueError("--onnx-file requires a trained agent and cannot be combined with checkpoint/comparison")
  if cfg.compare_onnx and dummy_mode:
    raise ValueError("ONNX comparison requires a trained checkpoint")
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg, actuator_info = prepare_env_cfg(task_id, cfg)
  from src.tasks.velocity.mdp.arm_intervention import ArmInterventionActionCfg
  arm_keyboard = cfg.keyboard_control and isinstance(env_cfg.actions.get("joint_pos"), ArmInterventionActionCfg)
  hand_keyboard = cfg.keyboard_control and "hand_tracking" in env_cfg.commands
  terminal_keyboard = hand_keyboard or arm_keyboard
  if cfg.keyboard_control:
    if cfg.headless_steps or cfg.viewer == "viser":
      raise ValueError("Keyboard controls require the native viewer; omit --headless-steps and use --viewer native")
    if not terminal_keyboard:
      _load_keyboard()
  agent_cfg = load_rl_cfg(task_id)
  DUMMY_MODE = dummy_mode
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task and cfg._demo_mode:
    env_cfg.commands["motion"].sampling_mode = "uniform"
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")


  log_dir = None
  resume_path = None
  if onnx_mode:
    onnx_path = Path(cfg.onnx_file)
    if not onnx_path.is_file():
      raise FileNotFoundError(onnx_path)
    log_dir = onnx_path.resolve().parent
  elif not dummy_mode:
    if cfg.checkpoint_file:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    elif cfg.wandb_run_path:
      log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      resume_path, _ = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
    else:
      raise ValueError("Provide --checkpoint-file, --onnx-file, or --wandb-run-path")
    log_dir = resume_path.parent

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device,
    render_mode="rgb_array" if cfg.video and not dummy_mode else None,
  )
  policy = None
  sim_logger = None
  try:
    if cfg.video and not dummy_mode:
      env = VideoRecorder(
        env, video_folder=log_dir / "videos" / "play",
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length, disable_logger=True,
      )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if dummy_mode:
      shape = env.unwrapped.action_space.shape
      if cfg.agent == "zero":
        policy = lambda obs: torch.zeros(shape, device=env.unwrapped.device)
      else:
        policy = lambda obs: 2 * torch.rand(shape, device=env.unwrapped.device) - 1
    elif onnx_mode:
      from scripts.onnx_sim_policy import OnnxSimPolicy
      policy = OnnxSimPolicy(cfg.onnx_file, action_dim=env.unwrapped.action_space.shape[-1])
    else:
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(env, asdict(agent_cfg), device=device)
      runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
      policy = runner.get_inference_policy(device=device)

    if cfg.compare_onnx:
      from scripts.mini_pawo_policy_compare import ComparedPolicy
      policy = ComparedPolicy(policy, env, cfg.compare_onnx, cfg.comparison_csv, resume_path)
    if cfg.sim_log:
      from scripts.mini_pawo_sim_logger import LoggedPolicy
      policy = LoggedPolicy(policy, env, filename=cfg.sim_log)
      sim_logger = policy
      import hashlib
      import json
      metadata = dict(
        task_id=task_id, action_scale=env_cfg.actions["joint_pos"].scale,
        clip_actions=agent_cfg.clip_actions, **actuator_info,
        physics_dt=float(env_cfg.sim.mujoco.timestep),
        step_dt=float(env.unwrapped.step_dt),
      )
      if onnx_mode:
        metadata.update(onnx=str(onnx_path.resolve()), sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest())
      Path(cfg.sim_log).with_suffix(".policy.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Refresh keyboard-dependent observation terms before the logger snapshots actor inputs.
    keyboard = None
    if cfg.keyboard_control:
      if arm_keyboard:
        from scripts.arm_intervention_control import KeyboardArmController
        keyboard = KeyboardArmController()
      elif hand_keyboard:
        from scripts.keyboard_hand_control import KeyboardHandController
        keyboard = KeyboardHandController()
      else:
        keyboard = KeyboardTwist()
      policy = CommandedPolicy(policy, env, keyboard)
    elif cfg.arm_control == "demo":
      from scripts.arm_intervention_control import ArmInterventionDemo
      policy = CommandedPolicy(policy, env, ArmInterventionDemo())

    if cfg.headless_steps:
      obs = env.get_observations()
      reset_count = 0
      min_height, max_tilt = float("inf"), 0.0
      with torch.inference_mode():
        for _ in range(cfg.headless_steps):
          obs, _, dones, _ = env.step(policy(obs))
          reset_count += int(dones.sum().item())
          robot = env.unwrapped.scene["robot"]
          min_height = min(min_height, float(robot.data.root_link_pos_w[:, 2].min()))
          quat = robot.data.root_link_quat_w
          upright_cos = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
          max_tilt = max(max_tilt, float(torch.acos(upright_cos.clamp(-1, 1)).max()) * 180 / 3.141592653589793)
          if hasattr(policy, "reset"):
            policy.reset(dones)
      import json
      summary = dict(task_id=task_id, seed=cfg.seed, steps=cfg.headless_steps,
                     resets=reset_count, min_root_height_m=min_height,
                     max_tilt_deg=max_tilt, **actuator_info)
      if cfg.sim_log:
        Path(cfg.sim_log).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str))
      print("[ROLLOUT]", summary)
      return

    viewer = cfg.viewer
    if viewer == "auto":
      display = sys.platform in {"darwin", "win32"} or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
      )
      viewer = "native" if display else "viser"
    if cfg.keyboard_control and viewer != "native":
      raise ValueError("Keyboard controls require --viewer native with a local display")
    if viewer == "native":
      if terminal_keyboard:
        keyboard.start()
      NativeMujocoViewer(env, policy).run()
    else:
      ViserPlayViewer(env, policy).run()
  finally:
    try:
      if hasattr(policy, "close"):
        policy.close()
    finally:
      try:
        env.close()
      finally:
        if sim_logger is not None and sim_logger.n > 0:
          if cfg.plot_actions:
            try:
              from scripts.plot_sim_policy import simulation, draw
              path = Path(cfg.sim_log)
              draw(simulation(path), None, path.parent / (path.stem + "_plots"))
            except Exception as exc:
              # Preserve playback errors and leave the CSV available for offline retry.
              print(f"[SIM PLOTS] Could not plot {cfg.sim_log}: {exc}")
          if cfg.plot_observations:
            try:
              from scripts.plot_sim_observations import draw_observations
              draw_observations(cfg.sim_log)
            except Exception as exc:
              print(f"[SIM OBSERVATIONS] Could not plot {cfg.sim_log}: {exc}")


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
