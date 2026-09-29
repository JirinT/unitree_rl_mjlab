"""Reach target command: samples a 3D end-effector target position."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["ReachCommand", "ReachCommandCfg"]


class ReachCommand(CommandTerm):
  """Samples a 3D target position (in the env-origin frame) for the end effector."""

  cfg: ReachCommandCfg

  def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.target_position_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    target = sample_uniform(lower, upper, (n, 3), device=self.device)
    self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=self.cfg.viz.radius,
        color=self.cfg.viz.target_color,
        label=f"reach_target_{batch}",
      )


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
  @dataclass
  class TargetPositionRangeCfg:
    """Target box in the env-origin frame (metres). Override per robot."""

    x: tuple[float, float] = (0.3, 0.6)
    y: tuple[float, float] = (-0.3, 0.3)
    z: tuple[float, float] = (0.2, 0.6)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.4)
    radius: float = 0.03

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
    return ReachCommand(self, env)
