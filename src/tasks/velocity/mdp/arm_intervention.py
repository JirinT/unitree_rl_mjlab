"""Episode-level arm takeover through real position-actuator targets (no IK).

Policy output/history are never mutated: PPO must retain its sampled actions.
Only the targets passed to the actuators are replaced. Public runtime API:
``set_external_targets`` / ``release_external_control`` / ``use_policy_control``.
"""

from dataclasses import dataclass, field
import math

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


ARM_JOINT_NAMES = tuple(
  f"{side}_{part}_joint"
  for side in ("left", "right")
  for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)


def _ranges():
  # Absolute joint radians; outward shoulders avoid intentionally crossing arms.
  return {
    ".*shoulder_pitch_joint": (-1.2, 0.6),
    "left_shoulder_roll_joint": (0.05, 1.05),
    "right_shoulder_roll_joint": (-1.05, -0.05),
    ".*shoulder_yaw_joint": (-0.45, 0.45),
    ".*elbow_joint": (0.15, 1.65),
    ".*wrist_roll_joint": (-0.5, 0.5),
    ".*wrist_pitch_joint": (-0.45, 0.45),
    ".*wrist_yaw_joint": (-0.4, 0.4),
  }


@dataclass(kw_only=True)
class ArmInterventionActionCfg(JointPositionActionCfg):
  # none, left, right, both; selected once at episode reset.
  mode_probabilities: tuple[float, float, float, float] = (0.5, 0.125, 0.125, 0.25)
  target_ranges: dict[str, tuple[float, float]] = field(default_factory=_ranges)
  takeover_delay_range: tuple[float, float] = (0.0, 2.0)
  intervention_time_range: tuple[float, float] = (10.0, 16.0)
  movement_time_range: tuple[float, float] = (1.5, 4.0)
  hold_time_range: tuple[float, float] = (0.3, 2.0)
  max_joint_speed: float = 1.2
  max_joint_acceleration: float = 3.0
  limit_margin: float = 0.03
  handover_time: float = 1.0
  difficulty: float = 1.0
  command_name: str = "twist"

  def __post_init__(self):
    super().__post_init__()
    if (len(self.mode_probabilities) != 4
        or any(not math.isfinite(x) or x < 0 for x in self.mode_probabilities)
        or not math.isclose(sum(self.mode_probabilities), 1.0)):
      raise ValueError("Expected four nonnegative mode probabilities summing to one.")
    for lo, hi in (self.takeover_delay_range, self.intervention_time_range, self.movement_time_range, self.hold_time_range, *self.target_ranges.values()):
      if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError("Trajectory ranges must be finite and ordered.")
    if (self.movement_time_range[0] <= 0 or self.hold_time_range[0] < 0
        or self.takeover_delay_range[0] < 0 or self.intervention_time_range[0] <= 0):
      raise ValueError("Movement times must be positive and holds nonnegative.")
    if any(not math.isfinite(x) or x <= 0 for x in (
      self.max_joint_speed, self.max_joint_acceleration, self.handover_time,
    )) or not math.isfinite(self.limit_margin) or self.limit_margin < 0:
      raise ValueError("Speed, acceleration and handover must be positive; margin nonnegative.")
    if not 0 < self.difficulty <= 1:
      raise ValueError("Difficulty must be in (0, 1].")

  def build(self, env):
    return ArmInterventionAction(self, env)


def smoothstep5(t):
  """Quintic interpolation, zero first and second derivatives at both ends."""
  return t**3 * (10 + t * (-15 + 6 * t))


class ArmInterventionAction(JointPositionAction):
  cfg: ArmInterventionActionCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    from mjlab.utils.lab_api.string import resolve_matching_names_values
    self.arm_joint_ids, names = self._entity.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    if tuple(names) != ARM_JOINT_NAMES or self.action_dim != 27:
      raise ValueError("Arm intervention requires the H1_2 full 27-joint position action.")
    indices = {name: i for i, name in enumerate(self.target_names)}
    self.arm_action_ids = torch.tensor([indices[n] for n in names], device=self.device)
    scale = torch.as_tensor(self._scale, device=self.device)
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
      raise ValueError("Intervention action scales must be finite and positive.")
    self.arm_default = self._entity.data.default_joint_pos[:, self.arm_joint_ids].clone()
    limits = self._entity.data.soft_joint_pos_limits[:, self.arm_joint_ids]
    self.lower = limits[..., 0] + cfg.limit_margin
    self.upper = limits[..., 1] - cfg.limit_margin
    if cfg.clip is not None:
      self.lower = torch.maximum(self.lower, self._clip[:, self.arm_action_ids, 0])
      self.upper = torch.minimum(self.upper, self._clip[:, self.arm_action_ids, 1])
    if bool((self.lower >= self.upper).any()):
      raise ValueError("Joint limit margin leaves no arm workspace.")
    ids, _, ranges = resolve_matching_names_values(cfg.target_ranges, names)
    if len(ids) != 14:
      raise ValueError("Target ranges must cover all 14 arm joints exactly once.")
    self.ranges = torch.zeros(14, 2, device=self.device)
    self.ranges[ids] = torch.tensor(ranges, device=self.device)
    self.masks = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
    self.planned_masks = torch.zeros_like(self.masks)
    self.episode_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.takeover_at = torch.zeros_like(self.episode_elapsed)
    self.release_at = torch.zeros_like(self.episode_elapsed)
    self.started = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.automatic = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
    self.releasing = torch.zeros_like(self.masks)
    self.release_elapsed = torch.zeros(self.num_envs, 2, device=self.device)
    self.release_start = self.arm_default.clone()
    self.reference = self.arm_default.clone()
    self.start = self.reference.clone()
    self.goal = self.reference.clone()
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.duration = torch.ones_like(self.elapsed)
    self.hold = torch.zeros_like(self.elapsed)
    self.executed_targets = self._entity.data.default_joint_pos[:, self.target_ids].clone()
    self.effective_actions = torch.zeros_like(self.raw_action)
    self.steps = torch.zeros_like(self.elapsed)
    self.linear_error_sum = torch.zeros_like(self.elapsed)
    self.yaw_error_sum = torch.zeros_like(self.elapsed)

  @property
  def control_mask(self):
    """Full takeover or release transition, left then right."""
    return self.masks | self.releasing

  @property
  def policy_joint_mask(self):
    """One for joints still controlled by PPO, in entity joint order."""
    weights = torch.ones_like(self._entity.data.joint_pos)
    weights[:, self.arm_joint_ids] = (~self.control_mask).repeat_interleave(7, dim=1).to(weights.dtype)
    return weights

  @property
  def policy_action_mask(self):
    weights = torch.ones_like(self.raw_action)
    weights[:, self.arm_action_ids] = (~self.control_mask).repeat_interleave(7, dim=1).to(weights.dtype)
    return weights

  def _ids(self, env_ids):
    all_ids = torch.arange(self.num_envs, device=self.device)
    return all_ids if env_ids is None else all_ids[env_ids]

  def _duration(self, start, goal, requested):
    delta = (goal - start).abs().amax(dim=1)
    # Exact maxima of the quintic's first/second derivatives: 1.875, 10/sqrt(3).
    speed = 1.875 * delta / self.cfg.max_joint_speed
    acceleration = torch.sqrt((10 / math.sqrt(3)) * delta / self.cfg.max_joint_acceleration)
    return torch.maximum(requested, torch.maximum(speed, acceleration))

  def _new_segment(self, ids, goal, requested):
    self.start[ids] = self.reference[ids]
    self.goal[ids] = torch.maximum(torch.minimum(goal, self.upper[ids]), self.lower[ids])
    self.duration[ids] = self._duration(self.start[ids], self.goal[ids], requested)
    self.elapsed[ids] = 0

  def _sample_segment(self, ids):
    if ids.numel() == 0:
      return
    sampled = self.ranges[:, 0] + torch.rand(len(ids), 14, device=self.device) * (self.ranges[:, 1] - self.ranges[:, 0])
    goal = self.arm_default[ids] + self.cfg.difficulty * (sampled - self.arm_default[ids])
    lo, hi = self.cfg.movement_time_range
    # Small curriculum amplitudes are also slower; no increase beyond configured speed.
    requested = torch.empty(len(ids), device=self.device).uniform_(lo, hi) / math.sqrt(self.cfg.difficulty)
    self._new_segment(ids, goal, requested)
    self.hold[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.hold_time_range)

  def reset(self, env_ids=None):
    ids = self._ids(env_ids)
    super().reset(ids)
    self.steps[ids] = self.linear_error_sum[ids] = self.yaw_error_sum[ids] = 0
    self.automatic[ids] = True
    self.releasing[ids] = False
    self.release_elapsed[ids] = 0
    modes = torch.multinomial(torch.tensor(self.cfg.mode_probabilities, device=self.device), len(ids), replacement=True)
    self.planned_masks[ids] = torch.stack((modes == 1, modes == 2), dim=1) | (modes == 3)[:, None]
    self.episode_elapsed[ids] = 0
    self.takeover_at[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.takeover_delay_range)
    self.release_at[ids] = self.takeover_at[ids] + torch.empty(len(ids), device=self.device).uniform_(*self.cfg.intervention_time_range)
    self.started[ids] = self.takeover_at[ids] <= 0
    self.masks[ids] = self.planned_masks[ids] & self.started[ids, None]
    initial = self._entity.data.joint_pos[ids][:, self.target_ids]
    self.executed_targets[ids] = initial
    self.reference[ids] = initial[:, self.arm_action_ids]
    self.effective_actions[ids] = 0  # Same initial history convention as the baseline.
    self._sample_segment(ids)

  def set_external_targets(self, targets, masks, *, env_ids=None, duration=1.5):
    """Schedule a deterministic absolute-radian waypoint; no random motion afterwards.

    Shapes: (selected envs, 14), (selected envs, 2), left then right in
    ARM_JOINT_NAMES order. Call when a waypoint changes, not every physics tick.
    A newly enabled arm starts at its last commanded target; release is blended.
    Requests outside soft limits are clamped. Duration grows to respect rate bounds.
    """
    ids = self._ids(env_ids)
    targets = torch.as_tensor(targets, device=self.device, dtype=self.reference.dtype)
    masks = torch.as_tensor(masks, device=self.device)
    if (targets.shape != (len(ids), 14) or masks.shape != (len(ids), 2)
        or not bool(torch.isfinite(targets).all())
        or not bool(((masks == 0) | (masks == 1)).all())
        or not math.isfinite(duration) or duration <= 0):
      raise ValueError("Expected finite (N,14) targets, binary (N,2) masks and positive duration.")
    masks = masks.bool()
    old = self.control_mask[ids]
    self.automatic[ids] = False
    # Start all paths from executed targets, not a stale inactive-arm reference.
    self.reference[ids] = self.executed_targets[ids][:, self.arm_action_ids]
    self.release_start[ids] = self.reference[ids]
    self.releasing[ids] = old & ~masks
    self.release_elapsed[ids] = 0
    self.masks[ids] = masks
    self._new_segment(ids, targets, torch.full((len(ids),), duration, device=self.device))
    self.hold[ids] = 0

  def release_external_control(self, env_ids=None):
    """Smoothly return both arms to the policy; disable random overrides until reset."""
    ids = self._ids(env_ids)
    self.set_external_targets(self.reference[ids].clone(), torch.zeros(len(ids), 2, device=self.device), env_ids=ids)

  def use_policy_control(self, env_ids=None):
    """Alias for a smooth release, also useful to disable automatic play trajectories."""
    self.release_external_control(env_ids)

  def process_actions(self, actions):
    super().process_actions(actions)
    dt = self._env.step_dt
    self.episode_elapsed += dt
    begin = (self.automatic & ~self.started & self.planned_masks.any(dim=1)
             & (self.episode_elapsed + 1e-6 >= self.takeover_at))
    ids = begin.nonzero(as_tuple=False).flatten()
    if ids.numel():
      self.started[ids] = True
      self.masks[ids] = self.planned_masks[ids]
      self.reference[ids] = self.executed_targets[ids][:, self.arm_action_ids]
      self._sample_segment(ids)
    finish = self.automatic & self.started & self.masks.any(dim=1) & (self.episode_elapsed + 1e-6 >= self.release_at)
    ids = finish.nonzero(as_tuple=False).flatten()
    if ids.numel():
      self.release_external_control(ids)
    resample = self.automatic & self.masks.any(dim=1) & (self.elapsed >= self.duration + self.hold)
    self._sample_segment(resample.nonzero(as_tuple=False).flatten())
    self.elapsed += dt
    t = (self.elapsed / self.duration).clamp(0, 1)
    self.reference[:] = torch.lerp(self.start, self.goal, smoothstep5(t)[:, None])
    policy_arms = self._processed_actions[:, self.arm_action_ids]
    target = torch.where(self.masks.repeat_interleave(7, dim=1), self.reference, policy_arms)
    self.release_elapsed += dt
    blend = smoothstep5((self.release_elapsed / self.cfg.handover_time).clamp(0, 1)).repeat_interleave(7, dim=1)
    release_target = torch.lerp(self.release_start, policy_arms, blend)
    target = torch.where(self.releasing.repeat_interleave(7, dim=1), release_target, target)
    previous = self.executed_targets[:, self.arm_action_ids]
    # Protect takeovers, repeated manual waypoints, and return-to-policy transitions.
    bound = self.cfg.max_joint_speed * dt
    bounded = previous + (target - previous).clamp(-bound, bound)
    bounded = torch.maximum(torch.minimum(bounded, self.upper), self.lower)
    controlled = self.control_mask.repeat_interleave(7, dim=1)
    self._processed_actions[:, self.arm_action_ids] = torch.where(controlled, bounded, policy_arms)
    self.executed_targets[:] = self._processed_actions
    self.effective_actions[:] = (self.executed_targets - self._offset) / self._scale
    close = ((bounded - policy_arms).abs().reshape(self.num_envs, 2, 7).amax(dim=2) < 1e-5)
    self.releasing &= ~((self.release_elapsed >= self.cfg.handover_time) & close)
    # Completed-episode curriculum statistics, consumed before action reset.
    command = self._env.command_manager.get_command(self.cfg.command_name)
    self.steps += self.control_mask.any(dim=1)
    active = self.control_mask.any(dim=1)
    self.linear_error_sum += ((self._entity.data.root_link_lin_vel_b[:, :2] - command[:, :2]).square().sum(dim=1)) * active
    self.yaw_error_sum += (self._entity.data.root_link_ang_vel_b[:, 2] - command[:, 2]).square() * active
