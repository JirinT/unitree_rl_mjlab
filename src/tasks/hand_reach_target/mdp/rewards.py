"""Reaching rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reach_command import ReachCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ee_position_kernel", "joint_velocity_hinge_penalty"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_position_kernel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-||ee - target||^2 / std^2)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(ReachCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  error = torch.sum(torch.square(ee_pos_w - command.target_pos), dim=-1)
  return torch.exp(-error / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared excess of |joint_vel| above max_vel (positive; use a negative weight)."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)
