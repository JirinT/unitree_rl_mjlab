"""Named combinations of changes to the Origami Pawo baseline."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from .modifiers.add_sensor_latency import add_sensor_latency
from .modifiers.add_actuator_latency import add_actuator_latency
from .modifiers.add_action_scale import add_action_scale

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

ACTUATOR_LATENCY = Recipe(
  name="actuator_latency",
  modifiers=(add_actuator_latency,),
)

ACTUATOR_SENSOR_LATENCY = Recipe(
  name="actuator_sensor_latency",
  modifiers=ACTUATOR_LATENCY.modifiers + SENSOR_LATENCY.modifiers,
)

ACTION_SCALE = Recipe(
  name="action_scale",
  modifiers=(add_action_scale,),
)
