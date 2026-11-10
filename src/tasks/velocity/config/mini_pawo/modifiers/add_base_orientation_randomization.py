"""Randomize initial roll/pitch to expose the policy to balance recovery."""

from copy import deepcopy
import math

from mjlab.envs import ManagerBasedRlEnvCfg

BASE_ORIENTATION_MAX_PITCH_DEG = 10.0
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0


def add_base_orientation_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add symmetric roll/pitch ranges to MiniPawo's existing root reset.

  reset_root_state_uniform samples each environment independently. Extending
  that event preserves its position, yaw, velocity and environment-origin
  handling and avoids a second reset event overwriting the sampled pose.
  Enabled in both training and playback for the recipes that select it.
  This changes initial conditions; interval pushes are a separate event.
  """
  limits = {
    "pitch": BASE_ORIENTATION_MAX_PITCH_DEG,
    "roll": BASE_ORIENTATION_MAX_ROLL_DEG,
  }
  for axis, value in limits.items():
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 90:
      raise ValueError(f"Maximum base {axis} must be finite and in [0, 90) degrees")
  if "reset_base" not in cfg.events:
    raise ValueError("Base orientation randomization requires the reset_base event")
  event = deepcopy(cfg.events["reset_base"])
  if event.mode != "reset" or not isinstance(event.params.get("pose_range"), dict):
    raise ValueError("reset_base must be a reset event with a pose_range dictionary")
  for axis, value in limits.items():
    angle = math.radians(value)
    event.params["pose_range"][axis] = (-angle, angle)
  cfg.events["reset_base"] = event
