"""Velocity tracking that tolerates external arm control; no Cartesian reaching reward."""

from copy import deepcopy
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from src.tasks.velocity.mdp.arm_intervention import ArmInterventionActionCfg
from src.tasks.velocity.mdp.arm_intervention_curriculum import arm_intervention_difficulty
from src.tasks.velocity.mdp.arm_intervention_rewards import (
  arm_control_state, contact_slip, executed_actions, intervention_action_rate,
  intervention_angular_momentum, intervention_joint_acc, intervention_metric,
  intervention_posture, intervention_stand_still,
)


def add_arm_intervention(cfg: ManagerBasedRlEnvCfg) -> None:
  if "hand_tracking" in cfg.commands:
    raise ValueError("Arm intervention is a separate recipe, not a hand-reaching modifier.")
  cfg.actions["joint_pos"] = ArmInterventionActionCfg(**asdict(cfg.actions["joint_pos"]))
  cfg.commands["twist"].rel_standing_envs = 0.5
  for name in ("actor", "critic"):
    cfg.observations[name] = deepcopy(cfg.observations[name])
    cfg.observations[name].terms["actions"].func = executed_actions
    cfg.observations[name].terms["arm_control"] = ObservationTermCfg(func=arm_control_state)
  replacements = {
    "pose": intervention_posture,
    "stand_still": intervention_stand_still,
    "action_rate_l2": intervention_action_rate,
    "joint_acc_l2": intervention_joint_acc,
    "angular_momentum": intervention_angular_momentum,
  }
  for name, function in replacements.items():
    cfg.rewards[name] = deepcopy(cfg.rewards[name])
    cfg.rewards[name].func = function
  # Lower body must be able to counter arm moments rather than freeze at rest.
  # Retain the existing standing arm preference whenever PPO owns that arm.
  cfg.rewards["pose"].params["std_standing"] = {
    ".*hip_yaw.*": 0.15, ".*hip_pitch.*": 0.3, ".*hip_roll.*": 0.15,
    ".*knee.*": 0.3, ".*ankle_pitch.*": 0.15, ".*ankle_roll.*": 0.1,
    ".*torso.*": 0.15, ".*shoulder.*": 0.05, ".*elbow.*": 0.05, ".*wrist.*": 0.05,
  }
  cfg.rewards["stand_still"].weight *= 0.25
  # Penalize only contact-foot motion, including at zero command.
  cfg.rewards["foot_slip"] = deepcopy(cfg.rewards["foot_slip"])
  cfg.rewards["foot_slip"].func = contact_slip
  cfg.curriculum["arm_intervention"] = CurriculumTermCfg(
    func=arm_intervention_difficulty,
    params={"initial_scale": 0.2, "scale_step": 0.2, "min_episodes": 128,
            "min_duration_s": 8., "success_fraction": 0.8,
            "linear_rmse": 0.25, "yaw_rmse": 0.5},
  )
  for name, kind in (("arm_intervention_fraction", "fraction"),
                     ("arm_intervention_difficulty", "difficulty"),
                     ("intervened_arm_joint_rmse", "arm_joint_rmse")):
    cfg.metrics[name] = MetricsTermCfg(func=intervention_metric, params={"kind": kind})
