"""Actor feedback from shared encoder measurements; privileged critic feedback."""

from copy import deepcopy

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.buffers import DelayBuffer
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg

from .hand_kinematics import PalmKinematics


def hand_positions(env, command_name: str):
  """Privileged palm positions from simulator state, left then right."""
  return env.command_manager.get_term(command_name).hand_positions_b.flatten(1)


def hand_target_errors(env, command_name: str):
  """Privileged masked errors for the critic; actor errors use encoder FK."""
  return env.command_manager.get_term(command_name).errors_b.flatten(1)


class actor_joint_hand_state:
  """One encoder sample supplies joint observations, palm FK, and target errors.

  Processing follows the original joint term: compute -> noise -> clip -> scale
  -> delay. FK undoes the observation scale and adds the default joint pose.
  Target commands/masks remain current, even when joint measurements are delayed.
  The outer observation manager applies history to the resulting shared packet.
  """

  def __init__(self, cfg, env):
    self.joint_cfg = deepcopy(cfg.params["joint_observation_cfg"])
    self.corrupt = cfg.params["enable_corruption"]
    self.robot = env.scene[env.command_manager.get_term(cfg.params["command_name"]).cfg.entity_name]
    for value in self.joint_cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    if self.joint_cfg.func(env, **self.joint_cfg.params).shape != self.robot.data.joint_pos.shape:
      raise ValueError("Shared hand feedback requires the full joint-position observation.")
    self.fk = PalmKinematics(
      self.robot.cfg.spec_fn().compile(), self.robot.joint_names,
      env.command_manager.get_term(cfg.params["command_name"]).cfg.site_names, env.device,
    )
    self.scale = torch.as_tensor(
      1.0 if self.joint_cfg.scale is None else self.joint_cfg.scale,
      dtype=torch.float32, device=env.device,
    )
    if not torch.isfinite(self.scale).all() or (self.scale == 0).any():
      raise ValueError("Joint observation scale must be finite and invertible for FK.")
    self.noise_model = None
    if self.corrupt and isinstance(self.joint_cfg.noise, NoiseModelCfg):
      self.noise_model = self.joint_cfg.noise.class_type(
        self.joint_cfg.noise, num_envs=env.num_envs, device=env.device,
      )
    self.delay = None
    if self.joint_cfg.delay_max_lag > 0:
      c = self.joint_cfg
      self.delay = DelayBuffer(
        min_lag=c.delay_min_lag, max_lag=c.delay_max_lag,
        batch_size=env.num_envs, device=env.device,
        per_env=c.delay_per_env, hold_prob=c.delay_hold_prob,
        update_period=c.delay_update_period, per_env_phase=c.delay_per_env_phase,
      )

  def reset(self, env_ids=None):
    if self.delay is not None:
      self.delay.reset(batch_ids=None if isinstance(env_ids, slice) else env_ids)
    if self.noise_model is not None:
      self.noise_model.reset(env_ids)

  def __call__(self, env, joint_observation_cfg, enable_corruption, command_name: str):
    del joint_observation_cfg, enable_corruption
    q = self.joint_cfg.func(env, **self.joint_cfg.params).clone()
    if self.corrupt and isinstance(self.joint_cfg.noise, NoiseCfg):
      q = self.joint_cfg.noise.apply(q)
    elif self.noise_model is not None:
      q = self.noise_model(q)
    if self.joint_cfg.clip is not None:
      q = q.clamp(*self.joint_cfg.clip)
    q = q * self.scale
    if self.delay is not None:
      self.delay.append(q)
      q = self.delay.compute()
    palms = self.fk(q / self.scale + self.robot.data.default_joint_pos)
    term = env.command_manager.get_term(command_name)
    errors = torch.where(term.masks[:, :, None].bool(), term.targets_b - palms, 0.0)
    return torch.cat((q, palms.flatten(1), errors.flatten(1)), dim=1)
