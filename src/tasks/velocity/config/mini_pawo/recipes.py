"""Named combinations of changes to the Mini Pawo baseline."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from .modifiers.add_identified_pwm_actuator import add_identified_pwm_actuator
from .modifiers.add_sensor_latency import add_sensor_latency
from .modifiers.add_bam_actuator import add_bam_actuator
from .modifiers.add_actuator_latency import add_actuator_latency
from .modifiers.add_action_scale import add_action_scale
from .modifiers.add_base_orientation_randomization import add_base_orientation_randomization

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

ACTUATOR_SENSOR_LATENCY = Recipe(
  name="actuator_sensor_latency",
  modifiers=ACTUATOR_LATENCY.modifiers + SENSOR_LATENCY.modifiers,
)

BAM_ACTUATOR_LATENCY = Recipe(
  name="bam_actuator_latency",
  modifiers=BAM.modifiers + ACTUATOR_LATENCY.modifiers,
)

BAM_SENSOR_ACTUATOR_LATENCY = Recipe(
  name="bam_sensor_actuator_latency",
  modifiers=BAM_SENSOR_LATENCY.modifiers + ACTUATOR_LATENCY.modifiers,
)

ACTION_SCALE = Recipe(
  name="action_scale",
  modifiers=(add_action_scale,),
)

BAM_ACTION_SCALE = Recipe(
  name="bam_action_scale",
  modifiers=BAM.modifiers + ACTION_SCALE.modifiers,
)


BAM_SENSOR_LATENCY_ACTION_SCALE = Recipe(
  name="bam_sensor_latency_action_scale",
  modifiers=BAM_SENSOR_LATENCY.modifiers + ACTION_SCALE.modifiers,
)

BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE = Recipe(
  name="bam_sensor_actuator_latency_action_scale",
  modifiers=BAM_SENSOR_ACTUATOR_LATENCY.modifiers + ACTION_SCALE.modifiers,
)

BASE_ORIENTATION_RANDOMIZATION = Recipe(
  name="base_orientation_randomization",
  modifiers=(add_base_orientation_randomization,),
)

ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION = Recipe(
  name="action_scale_base_orientation_randomization",
  modifiers=ACTION_SCALE.modifiers + BASE_ORIENTATION_RANDOMIZATION.modifiers,
)

BAM_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION = Recipe(
  name="bam_action_scale_base_orientation_randomization",
  modifiers=BAM_ACTION_SCALE.modifiers + BASE_ORIENTATION_RANDOMIZATION.modifiers,
)

BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION = Recipe(
  name="bam_sensor_actuator_latency_action_scale_base_orientation_randomization",
  modifiers=(BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE.modifiers
             + BASE_ORIENTATION_RANDOMIZATION.modifiers),
)


IDENTIFIED_PWM = Recipe(name="identified_pwm", modifiers=(add_identified_pwm_actuator,))
IDENTIFIED_PWM_ACTION_SCALE = Recipe(
    name="identified_pwm_action_scale", modifiers=IDENTIFIED_PWM.modifiers + ACTION_SCALE.modifiers,
)
IDENTIFIED_PWM_SENSOR_LATENCY_ACTION_SCALE = Recipe(
    name="identified_pwm_sensor_latency_action_scale",
    modifiers=IDENTIFIED_PWM_ACTION_SCALE.modifiers + SENSOR_LATENCY.modifiers,
)


IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY = Recipe(
    name="identified_pwm_sensor_actuator_latency",
    modifiers=IDENTIFIED_PWM.modifiers + SENSOR_LATENCY.modifiers + ACTUATOR_LATENCY.modifiers,
)
IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE = Recipe(
    name="identified_pwm_sensor_actuator_latency_action_scale",
    modifiers=IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY.modifiers + ACTION_SCALE.modifiers,
)
