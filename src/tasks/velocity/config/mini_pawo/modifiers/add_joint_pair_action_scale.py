"""Set symmetric left/right policy position-offset scales by joint pair."""

from mjlab.envs import ManagerBasedRlEnvCfg

JOINT_PAIR_ACTION_SCALES = {
  "hip_yaw": 0.5,
  "hip_roll": 0.5,
  "hip_pitch": 0.5,
  "knee": 0.5,
  "ankle": 0.5,
}


def add_joint_pair_action_scale(cfg: ManagerBasedRlEnvCfg) -> None:
  """
  Apply the same scale to each left/right pair, preserving HOME offsets.
  """
  pairs = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
  if set(JOINT_PAIR_ACTION_SCALES) != set(pairs):
    raise ValueError(f"Expected action scales for exactly these joint pairs: {pairs}")
  scales = {}
  for pair in pairs:
    value = JOINT_PAIR_ACTION_SCALES[pair]
    scales[f"(left|right)_{pair}"] = float(value)
  action = cfg.actions["joint_pos"]
  action.scale = scales
