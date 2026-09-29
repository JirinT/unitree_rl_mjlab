"""Reaching observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reach_command import ReachCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ee_to_target", "target_pos_rel_origin"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from the end-effector site to the target, world frame. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(ReachCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return command.target_pos - ee_pos_w


def target_pos_rel_origin(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Target position relative to the env origin. Shape (B, 3)."""
  command = cast(ReachCommand, env.command_manager.get_term(command_name))
  return command.target_pos - env.scene.env_origins
