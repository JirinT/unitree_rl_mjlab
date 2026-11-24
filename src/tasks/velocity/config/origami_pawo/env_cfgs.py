"""Build Origami Pawo environments from the baseline and an optional recipe."""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.origami_pawo.origami_pawo_constants import (
  ORIGAMI_PAWO_BUILTIN_ACTUATOR_HEAD,
  ORIGAMI_PAWO_BUILTIN_ACTUATOR_LEG,
)
from .baseline_cfg import origami_pawo_flat_env_cfg as make_baseline_flat_cfg
from .recipes import BASELINE, Recipe


def _apply_recipe(cfg: ManagerBasedRlEnvCfg, recipe: Recipe) -> ManagerBasedRlEnvCfg:
  # Isolate nested objects: robot constants and base templates can share them.
  cfg = deepcopy(cfg)
  robot = cfg.scene.entities["robot"]
  # The baseline always uses builtin servos, independently of the global BAM
  # default in origami_pawo_constants.py. Recipes can add command delays.
  robot.articulation = replace(
    robot.articulation,
    actuators=(
      deepcopy(ORIGAMI_PAWO_BUILTIN_ACTUATOR_LEG),
      deepcopy(ORIGAMI_PAWO_BUILTIN_ACTUATOR_HEAD),
    ),
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
