from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  mini_pawo_flat_env_cfg,
)

from dataclasses import replace

from .recipes import (
  IDENTIFIED_PWM, IDENTIFIED_PWM_ACTION_SCALE, IDENTIFIED_PWM_SENSOR_LATENCY_ACTION_SCALE,
  IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY,
  IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE,
  BAM_SENSOR_LATENCY_ACTION_SCALE,
  BASE_ORIENTATION_RANDOMIZATION,
  ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION,
  BAM_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION,
  BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION,
  BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE,
  ACTION_SCALE, 
  ACTUATOR_LATENCY, BASELINE, BAM, BAM_ACTUATOR_LATENCY, BAM_SENSOR_LATENCY,
  BAM_SENSOR_ACTUATOR_LATENCY, SENSOR_LATENCY, BAM_ACTION_SCALE, ACTUATOR_SENSOR_LATENCY
)
from .rl_cfg import mini_pawo_ppo_runner_cfg 

_TASK_RECIPES = (
  ("MiniPawo-Flat-IdentifiedPWM-SensorLatency-ActuatorLatency", IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY, "MiniPawo_identified_pwm_sensor_actuator_latency"),
  ("MiniPawo-Flat-IdentifiedPWM-SensorLatency-ActuatorLatency-ActionScale", IDENTIFIED_PWM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE, "MiniPawo_identified_pwm_sensor_actuator_latency_action_scale"),
  ("MiniPawo-Flat-IdentifiedPWM", IDENTIFIED_PWM, "MiniPawo_identified_pwm"),
  ("MiniPawo-Flat-IdentifiedPWM-ActionScale", IDENTIFIED_PWM_ACTION_SCALE, "MiniPawo_identified_pwm_action_scale"),
  ("MiniPawo-Flat-IdentifiedPWM-SensorLatency-ActionScale", IDENTIFIED_PWM_SENSOR_LATENCY_ACTION_SCALE, "MiniPawo_identified_pwm_sensor_latency_action_scale"),
  ("MiniPawo-Flat-BaseOrientationRandomization", BASE_ORIENTATION_RANDOMIZATION, "MiniPawo_base_orientation_randomization"),
  ("MiniPawo-Flat-ActionScale-BaseOrientationRandomization", ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION, "MiniPawo_action_scale_base_orientation"),
  ("MiniPawo-Flat-BAM-ActionScale-BaseOrientationRandomization", BAM_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION, "MiniPawo_bam_action_scale_base_orientation"),
  ("MiniPawo-Flat-BAM-SensorLatency-ActuatorLatency-ActionScale-BaseOrientationRandomization", BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE_BASE_ORIENTATION_RANDOMIZATION, "MiniPawo_bam_sensor_actuator_latency_action_scale_base_orientation"),
  ("MiniPawo-Flat-BAM-SensorLatency-ActuatorLatency-ActionScale", BAM_SENSOR_ACTUATOR_LATENCY_ACTION_SCALE, "MiniPawo_bam_sensor_actuator_latency_action_scale"),
  ("MiniPawo-Flat", BASELINE, "MiniPawo_velocity"),
  ("MiniPawo-Flat-ActionScale", ACTION_SCALE, "MiniPawo_action_scale"),
  ("MiniPawo-Flat-SensorLatency", SENSOR_LATENCY, "MiniPawo_sensor_latency"),
  ("MiniPawo-Flat-ActuatorSensorLatency", ACTUATOR_SENSOR_LATENCY, "MiniPawo_actuator_sensor_latency"),
  ("MiniPawo-Flat-ActuatorLatency", ACTUATOR_LATENCY, "MiniPawo_actuator_latency"),
  ("MiniPawo-Flat-BAM", BAM, "MiniPawo_bam"),
  ("MiniPawo-Flat-BAM-ActionScale", BAM_ACTION_SCALE, "MiniPawo_bam_action_scale"),
  ("MiniPawo-Flat-BAM-ActuatorLatency", BAM_ACTUATOR_LATENCY, "MiniPawo_bam_actuator_latency"),
  ("MiniPawo-Flat-BAM-SensorLatency-ActuatorLatency", BAM_SENSOR_ACTUATOR_LATENCY, "MiniPawo_bam_sensor_actuator_latency"),
  ("MiniPawo-Flat-BAM-SensorLatency", BAM_SENSOR_LATENCY, "MiniPawo_bam_sensor_latency"),
  ("MiniPawo-Flat-BAM-SensorLatency-ActionScale", BAM_SENSOR_LATENCY_ACTION_SCALE, "MiniPawo_bam_sensor_latency_action_scale"),
)

for task_id, recipe, experiment_name in _TASK_RECIPES:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=mini_pawo_flat_env_cfg(recipe=recipe),
    play_env_cfg=mini_pawo_flat_env_cfg(play=True, recipe=recipe),
    rl_cfg=replace(mini_pawo_ppo_runner_cfg(), experiment_name=experiment_name),
    runner_cls=VelocityOnPolicyRunner,
  )
