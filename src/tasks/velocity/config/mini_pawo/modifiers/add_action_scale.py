"""Use smaller policy joint-position offsets than the baseline."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

ACTION_SCALE = 0.5

def add_action_scale(cfg: ManagerBasedRlEnvCfg) -> None:
  """Scale each raw action by 0.25 rad, keeping the default-pose offset.

  This multiplies actions; it does not clip their values or change joint limits.
  The same scale is used in training and playback.
  """
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = ACTION_SCALE
