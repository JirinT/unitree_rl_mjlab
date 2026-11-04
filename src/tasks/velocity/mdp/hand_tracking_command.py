"""Position-only palm commands in the moving pelvis frame; no IK is used."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HandTrackingCommand(CommandTerm):
  cfg: HandTrackingCommandCfg

  def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.site_ids, _ = self.robot.find_sites(cfg.site_names, preserve_order=True)
    if len(self.site_ids) != 2:
      raise ValueError("Hand tracking requires exactly two palm sites, left then right.")
    # [left xyz, right xyz, left enabled, right enabled]. Inactive xyz is zero.
    self._command = torch.zeros(self.num_envs, 8, device=self.device)
    self.arm_joint_ids = [
      self.robot.find_joints(pattern)[0] for pattern in cfg.arm_joint_patterns
    ]
    if any(not ids for ids in self.arm_joint_ids):
      raise ValueError("Hand tracking requires joints for both arms.")
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.command_age = torch.zeros(self.num_envs, device=self.device)
    self.curriculum_samples = torch.zeros(self.num_envs, device=self.device)
    self.curriculum_successes = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def masks(self) -> torch.Tensor:
    return self._command[:, 6:8]

  @property
  def targets_b(self) -> torch.Tensor:
    return self._command[:, :6].reshape(self.num_envs, 2, 3)

  @property
  def hand_positions_b(self) -> torch.Tensor:
    relative_w = self.robot.data.site_pos_w[:, self.site_ids] - self.robot.data.root_link_pos_w[:, None]
    quat = self.robot.data.root_link_quat_w[:, None].expand(-1, 2, -1)
    return quat_apply_inverse(quat, relative_w)

  @property
  def errors_b(self) -> torch.Tensor:
    return torch.where(
      self.masks[:, :, None].bool(), self.targets_b - self.hand_positions_b, 0.0
    )

  @property
  def posture_mask(self) -> torch.Tensor:
    """Keep posture regularization except on each actively tracking arm."""
    weights = torch.ones(self.num_envs, len(self.robot.joint_names), device=self.device)
    for side, joint_ids in enumerate(self.arm_joint_ids):
      weights[:, joint_ids] = 1.0 - self.masks[:, side:side + 1]
    return weights

  def set_targets(
    self, targets_b: torch.Tensor, masks: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Set pelvis-frame targets in metres and binary masks for selected envs.

    Commands otherwise resample on the normal timer. For external control, set
    resampling_time_range to a long interval and call this after env.reset().
    """
    ids = slice(None) if env_ids is None else env_ids
    count = self.num_envs if env_ids is None else len(env_ids)
    targets_b = torch.as_tensor(targets_b, dtype=self._command.dtype, device=self.device)
    masks = torch.as_tensor(masks, dtype=self._command.dtype, device=self.device)
    if targets_b.shape != (count, 2, 3) or masks.shape != (count, 2):
      raise ValueError("Expected targets (N, 2, 3) and masks (N, 2).")
    if not torch.isfinite(targets_b).all() or not ((masks == 0) | (masks == 1)).all():
      raise ValueError("Targets must be finite and masks must be binary.")
    self._command[ids, :6] = torch.where(masks[:, :, None].bool(), targets_b, 0.0).flatten(1)
    self._command[ids, 6:8] = masks
    self.command_age[ids] = 0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    targets = torch.empty(count, 2, 3, device=self.device)
    for side, ranges in enumerate((self.cfg.left_ranges, self.cfg.right_ranges)):
      for axis, bounds in enumerate(ranges):
        center = (bounds[0] + bounds[1]) * 0.5
        half_width = (bounds[1] - bounds[0]) * 0.5 * self.cfg.workspace_scale
        targets[:, side, axis].uniform_(center - half_width, center + half_width)
    probabilities = torch.tensor(self.cfg.mode_probabilities, device=self.device)
    modes = torch.multinomial(probabilities, count, replacement=True)
    # none, left only, right only, both; sampled independently of twist.
    masks = torch.stack(((modes == 1) | (modes == 3), (modes == 2) | (modes == 3)), dim=1)
    self.set_targets(targets, masks, env_ids)
    self.command_age[env_ids] = 0

  def _update_command(self) -> None:
    pass  # Targets remain fixed relative to the moving pelvis until resampled.

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    self.curriculum_samples[env_ids] = 0
    self.curriculum_successes[env_ids] = 0
    return extras

  def _update_metrics(self) -> None:
    self.command_age += self._env.step_dt
    eligible = self.masks.bool() & (self.command_age >= self.cfg.curriculum_settle_s)[:, None]
    errors = self.errors_b.norm(dim=-1)
    self.curriculum_samples += eligible.sum(dim=1)
    self.curriculum_successes += ((errors <= self.cfg.curriculum_tolerance) & eligible).sum(dim=1)
    count = self.masks.sum(dim=1).clamp_min(1)
    error = self.errors_b.norm(dim=-1).sum(dim=1) / count
    self.metrics["position_error"] += error * self._env.step_dt / self.cfg.resampling_time_range[1]

  def _debug_vis_impl(self, visualizer) -> None:
    quat = self.robot.data.root_link_quat_w[:, None].expand(-1, 2, -1)
    targets_w = self.robot.data.root_link_pos_w[:, None] + quat_apply(quat, self.targets_b)
    for idx in visualizer.get_env_indices(self.num_envs):
      for side, color in enumerate(((0.2, 0.8, 0.2, 0.8), (0.2, 0.4, 1.0, 0.8))):
        if self.masks[idx, side] > 0:
          visualizer.add_sphere(targets_w[idx, side], radius=0.035, color=color)


@dataclass(kw_only=True)
class HandTrackingCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  site_names: tuple[str, str] = ("left_palm", "right_palm")
  arm_joint_patterns: tuple[str, str] = (
    r"left_(shoulder|elbow|wrist).*", r"right_(shoulder|elbow|wrist).*",
  )
  # x forward, y left, z up in pelvis coordinates (metres).
  left_ranges: tuple[tuple[float, float], ...] = ((0.15, 0.45), (0.15, 0.40), (0.0, 0.35))
  right_ranges: tuple[tuple[float, float], ...] = ((0.15, 0.45), (-0.40, -0.15), (0.0, 0.35))
  mode_probabilities: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

  # Fraction of each final box's half-width, about its midpoint. Playback uses 1.
  workspace_scale: float = 1.0
  curriculum_tolerance: float = 0.10
  curriculum_settle_s: float = 0.5

  def __post_init__(self):
    import math

    if not math.isfinite(self.workspace_scale) or not 0 < self.workspace_scale <= 1:
      raise ValueError("Workspace scale must lie in (0, 1].")
    if (not math.isfinite(self.curriculum_tolerance) or self.curriculum_tolerance <= 0
        or not math.isfinite(self.curriculum_settle_s) or self.curriculum_settle_s < 0):
      raise ValueError("Curriculum tolerance must be positive and settle time nonnegative.")
    if len(self.site_names) != 2 or len(self.arm_joint_patterns) != 2:
      raise ValueError("Configure two hand sites and two arm joint patterns.")
    if (len(self.mode_probabilities) != 4
        or any(not math.isfinite(p) or p < 0 for p in self.mode_probabilities)
        or not math.isclose(sum(self.mode_probabilities), 1.0)):
      raise ValueError("Four finite, nonnegative mode probabilities must sum to one.")
    if (len(self.resampling_time_range) != 2
        or not 0 < self.resampling_time_range[0] <= self.resampling_time_range[1]
        or not all(math.isfinite(v) for v in self.resampling_time_range)):
      raise ValueError("Resampling times must be finite, ordered, and positive.")
    for ranges in (self.left_ranges, self.right_ranges):
      if len(ranges) != 3 or any(
        len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]
        for bounds in ranges
      ):
        raise ValueError("Hand ranges require three finite, ordered (min, max) pairs.")

  def build(self, env: ManagerBasedRlEnv) -> HandTrackingCommand:
    return HandTrackingCommand(self, env)
