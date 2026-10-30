"""Named combinations of changes to the Mini Pawo baseline."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from .modifiers.add_sensor_latency import add_sensor_latency
from .modifiers.add_bam_actuator import add_bam_actuator
from .modifiers.add_actuator_latency import add_actuator_latency

# A modifier edits its own environment configuration in place.
Modifier = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class Recipe:
  name: str
  modifiers: tuple[Modifier, ...] = ()


BASELINE = Recipe(name="baseline")

SENSOR_LATENCY = Recipe(
  name="sensor_latency",
  modifiers=(add_sensor_latency,),
)

BAM = Recipe(
  name="bam",
  modifiers=(add_bam_actuator,),
)

BAM_SENSOR_LATENCY = Recipe(
  name="bam_sensor_latency",
  modifiers=BAM.modifiers + SENSOR_LATENCY.modifiers,
)

ACTUATOR_LATENCY = Recipe(
  name="actuator_latency",
  modifiers=(add_actuator_latency,),
)

BAM_ACTUATOR_LATENCY = Recipe(
  name="bam_actuator_latency",
  modifiers=BAM.modifiers + ACTUATOR_LATENCY.modifiers,
)

BAM_SENSOR_ACTUATOR_LATENCY = Recipe(
  name="bam_sensor_actuator_latency",
  modifiers=BAM_SENSOR_LATENCY.modifiers + ACTUATOR_LATENCY.modifiers,
)
