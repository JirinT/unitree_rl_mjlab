"""Build H1_2 environments from the baseline and an optional recipe."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .baseline_cfg import (
  unitree_h1_2_flat_env_cfg as make_baseline_flat_cfg,
  unitree_h1_2_rough_env_cfg as make_baseline_rough_cfg,
)
from .recipes import BASELINE, Recipe


def _apply_recipe(cfg: ManagerBasedRlEnvCfg, recipe: Recipe, *, play: bool = False) -> ManagerBasedRlEnvCfg:
  # Isolate nested objects: robot constants and base templates can share them.
  cfg = deepcopy(cfg)
  for modifier in recipe.modifiers:
    modifier(cfg)
  if play:
    # Recipe modifiers may add curricula after baseline playback overrides.
    cfg.curriculum.clear()
  return cfg


def unitree_h1_2_rough_env_cfg(
  play: bool = False, *, recipe: Recipe = BASELINE,
) -> ManagerBasedRlEnvCfg:
  """Rough baseline by default; apply recipe modifiers in their listed order."""
  return _apply_recipe(make_baseline_rough_cfg(play=play), recipe, play=play)


def unitree_h1_2_flat_env_cfg(
  play: bool = False, *, recipe: Recipe = BASELINE,
) -> ManagerBasedRlEnvCfg:
  """Flat baseline by default; apply recipe modifiers in their listed order."""
  return _apply_recipe(make_baseline_flat_cfg(play=play), recipe, play=play)
