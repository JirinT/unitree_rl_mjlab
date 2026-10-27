"""Build Mini Pawo environments from the baseline and an optional recipe."""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.mini_pawo.minipawo_constants import (
  MINI_PAWO_BUILTIN_ACTUATOR_LEG,
)
from .baseline_cfg import mini_pawo_flat_env_cfg as make_baseline_flat_cfg
from .recipes import BASELINE, Recipe


def _apply_recipe(cfg: ManagerBasedRlEnvCfg, recipe: Recipe) -> ManagerBasedRlEnvCfg:
  # Isolate nested objects: robot constants and base templates can share them.
  cfg = deepcopy(cfg)
  robot = cfg.scene.entities["robot"]
  # The baseline always uses builtin servos, independently of the global BAM
  # default in minipawo_constants.py. No command-delay wrapper is added.
  robot.articulation = replace(
    robot.articulation,
    actuators=(deepcopy(MINI_PAWO_BUILTIN_ACTUATOR_LEG),),
  )
  cfg.events.pop("bam_init", None)

  for modifier in recipe.modifiers:
    modifier(cfg)
  return cfg


def mini_pawo_flat_env_cfg(
  play: bool = False, *, recipe: Recipe = BASELINE,
) -> ManagerBasedRlEnvCfg:
  """Flat baseline by default; apply recipe modifiers in their listed order."""
  return _apply_recipe(make_baseline_flat_cfg(play=play), recipe)
