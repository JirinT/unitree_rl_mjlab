"""Increase arm disturbance amplitude only after successful locomotion episodes."""

import torch


class arm_intervention_difficulty:
  def __init__(self, cfg, env):
    self.term = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
    p = cfg.params
    start, step = p.get("initial_scale", 0.2), p.get("scale_step", 0.2)
    if not 0 < start <= 1 or not 0 < step <= 1:
      raise ValueError("Curriculum scales must lie in (0, 1].")
    if (p.get("min_episodes", 128) < 1 or p.get("min_duration_s", 8.) <= 0
        or not 0 < p.get("success_fraction", 0.8) <= 1
        or p.get("linear_rmse", 0.25) <= 0 or p.get("yaw_rmse", 0.5) <= 0):
      raise ValueError("Invalid intervention curriculum thresholds.")
    self.term.cfg.difficulty = start
    self.episodes = self.successes = 0

  def __call__(self, env, env_ids, action_name: str = "joint_pos", initial_scale: float = 0.2,
               scale_step: float = 0.2, min_episodes: int = 128, min_duration_s: float = 8.,
               success_fraction: float = 0.8, linear_rmse: float = 0.25, yaw_rmse: float = 0.5):
    del action_name, initial_scale
    ids = slice(None) if env_ids is None else env_ids
    steps = self.term.steps[ids]
    eligible = steps > 0
    succeeded = (~env.termination_manager.terminated[ids]
                 & (steps * env.step_dt >= min_duration_s)
                 & (self.term.linear_error_sum[ids] / steps.clamp_min(1) <= linear_rmse**2)
                 & (self.term.yaw_error_sum[ids] / steps.clamp_min(1) <= yaw_rmse**2))
    self.episodes += int(eligible.sum().item())
    self.successes += int((eligible & succeeded).sum().item())
    # Consume once before action reset; non-intervention episodes cannot promote.
    self.term.steps[ids] = 0
    self.term.linear_error_sum[ids] = 0
    self.term.yaw_error_sum[ids] = 0
    if self.episodes >= min_episodes:
      if self.successes / self.episodes >= success_fraction:
        self.term.cfg.difficulty = min(1.0, round(self.term.cfg.difficulty + scale_step, 10))
      self.episodes = self.successes = 0
    return torch.tensor(self.term.cfg.difficulty, device=env.device)
