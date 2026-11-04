"""Performance-gated expansion of hand target boxes, using completed episodes."""

import torch


class hand_tracking_workspace:
  """Expand both boxes about their centers after successful evaluation windows.

  Disabled hands and the post-command settling interval supply no samples.
  Survival uses completed episodes with tracking samples; no-reaching episodes
  cannot advance the curriculum. Failed windows retry at the current scale.
  """

  def __init__(self, cfg, env):
    p = cfg.params
    self.term = env.command_manager.get_term(p["command_name"])
    start, step = p["initial_scale"], p["scale_step"]
    if not 0 < start <= 1 or not 0 < step <= 1:
      raise ValueError("Workspace scales must be in (0, 1].")
    if p["min_episodes"] < 1 or p["min_samples"] < 1 or p["min_survival_s"] < 0:
      raise ValueError("Curriculum sample counts must be positive and survival nonnegative.")
    if not 0 <= p["success_fraction"] <= 1 or not 0 <= p["survival_fraction"] <= 1:
      raise ValueError("Curriculum fractions must lie in [0, 1].")
    self.term.cfg.workspace_scale = start
    self.episodes = 0
    self.survived = 0
    self.samples = 0.0
    self.successes = 0.0

  def __call__(
    self, env, env_ids, command_name: str, initial_scale: float = 0.2,
    scale_step: float = 0.2, success_fraction: float = 0.8,
    survival_fraction: float = 0.8, min_episodes: int = 128,
    min_samples: int = 512, min_survival_s: float = 5.0,
  ):
    del command_name, initial_scale
    ids = slice(None) if env_ids is None else env_ids
    counts = self.term.curriculum_samples[ids]
    eligible = counts > 0
    self.episodes += int(eligible.sum().item())
    survived = (~env.termination_manager.terminated[ids]
                & (env.episode_length_buf[ids] * env.step_dt >= min_survival_s))
    self.survived += int((eligible & survived).sum().item())
    self.samples += counts.sum().item()
    # Samples from fallen/short episodes count as failures for promotion.
    self.successes += (self.term.curriculum_successes[ids] * survived).sum().item()
    # Consume once; command reset also clears these after curriculum.compute().
    self.term.curriculum_samples[ids] = 0
    self.term.curriculum_successes[ids] = 0
    if self.episodes >= min_episodes and self.samples >= min_samples:
      if (self.successes / self.samples >= success_fraction
          and self.survived / self.episodes >= survival_fraction):
        self.term.cfg.workspace_scale = min(1.0, round(self.term.cfg.workspace_scale + scale_step, 10))
      self.episodes = self.survived = 0
      self.samples = self.successes = 0.0
    return torch.tensor(self.term.cfg.workspace_scale, device=env.device)
