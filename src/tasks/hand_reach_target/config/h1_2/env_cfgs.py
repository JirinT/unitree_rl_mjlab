"""H1_2 left-arm reaching: fills in the per-robot fields of the reach task."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.unitree_h1_2_hand.h1_2_hand_constants import (
  EE_SITE_NAME,
  H1_2_LEFT_ARM_ACTION_SCALE,
  get_h1_2_left_arm_robot_cfg,
)
from src.tasks.hand_reach_target.mdp import ReachCommandCfg
from src.tasks.hand_reach_target.reach_env_cfg import make_reach_env_cfg


def unitree_h1_2_left_arm_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_reach_env_cfg()

  cfg.scene.entities = {"robot": get_h1_2_left_arm_robot_cfg()}
  cfg.scene.env_spacing = 2.5

  # Action scale (dict keyed by joint name).
  cfg.actions["joint_pos"].scale = H1_2_LEFT_ARM_ACTION_SCALE

  # End-effector site: left_palm (defined in h1_2.xml).
  ee = (EE_SITE_NAME,)
  cfg.observations["actor"].terms["ee_to_target"].params["asset_cfg"].site_names = ee
  cfg.observations["critic"].terms["ee_to_target"].params["asset_cfg"].site_names = ee
  cfg.rewards["reach_coarse"].params["asset_cfg"].site_names = ("left_palm",)
  cfg.rewards["reach_fine"].params["asset_cfg"].site_names = ("left_palm",)

  cfg.commands["reach_target"].target_position_range = (
    ReachCommandCfg.TargetPositionRangeCfg(
      x=(0.2, 0.45),
      y=(0.10, 0.40),
      z=(1.20, 1.50),
    )
  )

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
