"""Named combinations of changes to the Mini Pawo baseline."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

# A modifier edits its own environment configuration in place.
Modifier = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class Recipe:
  name: str
  modifiers: tuple[Modifier, ...] = ()


BASELINE = Recipe(name="baseline")

# Future recipes can combine modifiers without copying the baseline:
# BAM_DELAYS = Recipe(name="bam_delays", modifiers=(apply_bam, apply_delays))
