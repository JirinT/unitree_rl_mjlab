"""Add optional palm position tracking while retaining full-body joint actions."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.velocity.mdp.hand_tracking_command import HandTrackingCommandCfg
from src.tasks.velocity.mdp.hand_tracking_curriculum import hand_tracking_workspace
from src.tasks.velocity.mdp.hand_tracking_observations import (
  actor_joint_hand_state,
  hand_positions,
  hand_target_errors,
)
from src.tasks.velocity.mdp.hand_tracking_rewards import (
  hand_tracking_posture,
  hand_tracking_stand_still,
  track_hand_positions,
)
from src.tasks.velocity.mdp import generated_commands


def add_hand_tracking(cfg: ManagerBasedRlEnvCfg) -> None:
  """Compose shared encoder feedback, masked reaching, and workspace expansion."""
  cfg.commands["hand_tracking"] = HandTrackingCommandCfg(
    resampling_time_range=(3.0, 8.0), debug_vis=True,
  )
  actor = cfg.observations["actor"]
  joints = deepcopy(actor.terms["joint_pos"])
  shared_state = ObservationTermCfg(
    func=actor_joint_hand_state,
    params={
      "joint_observation_cfg": joints,
      "enable_corruption": actor.enable_corruption,
      "command_name": "hand_tracking",
    },
    history_length=joints.history_length,
    flatten_history_dim=joints.flatten_history_dim,
  )
  actor.terms = {
    ("joint_hand_state" if name == "joint_pos" else name):
      (shared_state if name == "joint_pos" else term)
    for name, term in actor.terms.items()
  }
  actor.terms["hand_command"] = ObservationTermCfg(
    func=generated_commands, params={"command_name": "hand_tracking"},
  )
  cfg.observations["critic"].terms.update({
    "hand_command": ObservationTermCfg(
      func=generated_commands, params={"command_name": "hand_tracking"},
    ),
    "hand_positions": ObservationTermCfg(
      func=hand_positions, params={"command_name": "hand_tracking"},
    ),
    "hand_target_errors": ObservationTermCfg(
      func=hand_target_errors, params={"command_name": "hand_tracking"},
    ),
  })
  cfg.rewards["track_hand_positions"] = RewardTermCfg(
    func=track_hand_positions, weight=2.0,
    params={"command_name": "hand_tracking", "std": 0.15},
  )
  for name, func in (("pose", hand_tracking_posture), ("stand_still", hand_tracking_stand_still)):
    cfg.rewards[name] = deepcopy(cfg.rewards[name])
    cfg.rewards[name].func = func
    cfg.rewards[name].params["hand_command_name"] = "hand_tracking"
  cfg.curriculum["hand_workspace"] = CurriculumTermCfg(
    func=hand_tracking_workspace,
    params={
      "command_name": "hand_tracking",
      "initial_scale": 0.2,
      "scale_step": 0.2,
      "success_fraction": 0.8,
      "survival_fraction": 0.8,
      "min_episodes": 128,
      "min_samples": 512,
      "min_survival_s": 5.0,
    },
  )
