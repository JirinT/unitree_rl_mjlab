"""Regularization respects which joints the policy can actually control."""

import torch

from .rewards import variable_posture


def _term(env, action_name):
  return env.action_manager.get_term(action_name)


def executed_actions(env, action_name: str = "joint_pos"):
  """Previous executed target, in policy scale/offset units; never sampled fiction."""
  return _term(env, action_name).effective_actions


def arm_control_state(env, action_name: str = "joint_pos"):
  """[left/right takeover, 14 executed arm targets relative to default], radians.

  Inactive targets are zero. Measured joint observations retain baseline noise.
  These references are known commands, not privileged simulator measurements.
  """
  term = _term(env, action_name)
  mask = term.control_mask
  relative = term.executed_targets[:, term.arm_action_ids] - term.arm_default
  return torch.cat((mask.float(), relative * mask.repeat_interleave(7, dim=1)), dim=1)


class intervention_posture(variable_posture):
  def __call__(self, env, std_standing, std_walking, std_running, asset_cfg,
               command_name: str, action_name: str = "joint_pos",
               walking_threshold: float = 0.5, running_threshold: float = 1.5):
    del std_standing, std_walking, std_running
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
    std = torch.where((speed < walking_threshold)[:, None], self.std_standing,
                      torch.where((speed < running_threshold)[:, None], self.std_walking, self.std_running))
    errors = (asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]).square() / std.square()
    weights = _term(env, action_name).policy_joint_mask[:, asset_cfg.joint_ids]
    # Normalize by remaining joints so a moving arm cannot dilute leg posture cost.
    return torch.exp(-(errors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))


def intervention_stand_still(env, command_name: str, asset_cfg, command_threshold: float = 0.1,
                            action_name: str = "joint_pos"):
  asset = env.scene[asset_cfg.name]
  error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).square()
  weights = _term(env, action_name).policy_joint_mask[:, asset_cfg.joint_ids]
  command = env.command_manager.get_command(command_name)
  standing = command[:, :2].norm(dim=1) + command[:, 2].abs() <= command_threshold
  return (error * weights).sum(dim=1) * standing


def intervention_action_rate(env, action_name: str = "joint_pos"):
  # Raw PPO actions remain intact for probability ratios and rollout storage.
  delta = env.action_manager.action - env.action_manager.prev_action
  return (delta.square() * _term(env, action_name).policy_action_mask).sum(dim=1)


def intervention_joint_acc(env, action_name: str = "joint_pos"):
  term = _term(env, action_name)
  return (term._entity.data.joint_acc.square() * term.policy_joint_mask).sum(dim=1)


def intervention_angular_momentum(env, sensor_name: str, action_name: str = "joint_pos"):
  # Whole-body momentum includes externally prescribed arms. Keep upright/torso
  # angular-velocity penalties active; don't require cancelling every arm motion.
  free = ~_term(env, action_name).control_mask.any(dim=1)
  return env.scene[sensor_name].data.square().sum(dim=-1) * free


def intervention_metric(env, kind: str, action_name: str = "joint_pos"):
  term = _term(env, action_name)
  if kind == "fraction":
    return term.control_mask.any(dim=1).float()
  if kind == "difficulty":
    return torch.full((env.num_envs,), term.cfg.difficulty, device=env.device)
  if kind == "arm_joint_rmse":
    weights = term.control_mask.repeat_interleave(7, dim=1)
    error = term._entity.data.joint_pos[:, term.arm_joint_ids] - term.executed_targets[:, term.arm_action_ids]
    return torch.sqrt((error.square() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))
  raise ValueError(f"Unknown intervention metric: {kind}")


def contact_slip(env, sensor_name: str, command_name: str, asset_cfg,
                 command_threshold: float = 0.1):
  """Contact-aware foot slip during both walking and zero-command standing."""
  del command_name, command_threshold
  sensor = env.scene[sensor_name]
  asset = env.scene[asset_cfg.name]
  contact = (sensor.data.found > 0).float()
  speed_sq = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].square().sum(dim=-1)
  return (speed_sq * contact).sum(dim=1)
