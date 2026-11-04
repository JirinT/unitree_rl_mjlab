"""Masked hand tracking and posture rewards for whole-body PPO control."""

import torch

from .rewards import variable_posture


def track_hand_positions(env, command_name: str, std: float = 0.15):
  """Mean exponential reward over active hands; exactly zero if both are off."""
  if std <= 0:
    raise ValueError("Hand tracking std must be positive.")
  term = env.command_manager.get_term(command_name)
  scores = torch.exp(-term.errors_b.square().sum(dim=-1) / std**2)
  return (scores * term.masks).sum(dim=1) / term.masks.sum(dim=1).clamp_min(1)


class hand_tracking_posture(variable_posture):
  """Speed-dependent baseline posture, freeing only actively tracking arms."""

  def __call__(
    self, env, std_standing, std_walking, std_running, asset_cfg,
    command_name: str, hand_command_name: str,
    walking_threshold: float = 0.5, running_threshold: float = 1.5,
  ):
    del std_standing, std_walking, std_running
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
    std = torch.where(
      (speed < walking_threshold)[:, None], self.std_standing,
      torch.where((speed < running_threshold)[:, None], self.std_walking, self.std_running),
    )
    errors = (asset.data.joint_pos[:, asset_cfg.joint_ids]
              - self.default_joint_pos[:, asset_cfg.joint_ids]).square() / std.square()
    weights = env.command_manager.get_term(hand_command_name).posture_mask[:, asset_cfg.joint_ids]
    return torch.exp(-(errors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))


def hand_tracking_stand_still(
  env, command_name: str, hand_command_name: str, asset_cfg,
  command_threshold: float = 0.1,
):
  """Keep baseline standing cost without pulling active arms to their rest pose."""
  asset = env.scene[asset_cfg.name]
  errors = (asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).square()
  weights = env.command_manager.get_term(hand_command_name).posture_mask[:, asset_cfg.joint_ids]
  command = env.command_manager.get_command(command_name)
  speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
  return (errors * weights).sum(dim=1) * (speed <= command_threshold)
