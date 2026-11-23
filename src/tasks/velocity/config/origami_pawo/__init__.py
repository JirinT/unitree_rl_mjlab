from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import mini_pawo_flat_env_cfg
from .recipes import (
  ACTION_SCALE,
  ACTUATOR_LATENCY,
  ACTUATOR_SENSOR_LATENCY,
  BASELINE,
  SENSOR_LATENCY,
)
from .rl_cfg import mini_pawo_ppo_runner_cfg

TASK_RECIPES = (
  ("OrigamiPawo-Flat-Baseline", BASELINE, "OrigamiPawo_baseline"),
  ("OrigamiPawo-Flat-ActionScale", ACTION_SCALE, "OrigamiPawo_action_scale"),
  ("OrigamiPawo-Flat-SensorLatency", SENSOR_LATENCY, "OrigamiPawo_sensor_latency"),
  ("OrigamiPawo-Flat-ActuatorSensorLatency", ACTUATOR_SENSOR_LATENCY, "OrigamiPawo_actuator_sensor_latency"),
  ("OrigamiPawo-Flat-ActuatorLatency", ACTUATOR_LATENCY, "OrigamiPawo_actuator_latency"),
)

for task_id, recipe, experiment_name in TASK_RECIPES:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=mini_pawo_flat_env_cfg(recipe=recipe),
    play_env_cfg=mini_pawo_flat_env_cfg(play=True, recipe=recipe),
    rl_cfg=replace(mini_pawo_ppo_runner_cfg(), experiment_name=experiment_name),
    runner_cls=VelocityOnPolicyRunner,
  )
