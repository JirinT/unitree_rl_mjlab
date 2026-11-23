"""One uniform policy joint-position offset scale for all MiniPawo joints."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

# EDIT THIS: radians of position offset per raw policy action unit.
# Shared by every ActionScale recipe, including legacy task-name aliases.
ACTION_SCALE = 0.25


def add_action_scale(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the joint_pos scale with one scalar, preserving HOME offsets.

  Training and playback use the same scale. Zero removes all policy position
  offsets. Scaling does not clip actions/targets or change servo gains.
  """
  if not isinstance(ACTION_SCALE, (int, float)) or not math.isfinite(ACTION_SCALE) or ACTION_SCALE < 0:
    raise ValueError("ACTION_SCALE must be finite and nonnegative")
  action = cfg.actions["joint_pos"]
  if not isinstance(action, JointPositionActionCfg):
    raise TypeError("Action scaling requires a JointPositionActionCfg")
  action.scale = float(ACTION_SCALE)
