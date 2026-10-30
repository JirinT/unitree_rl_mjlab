from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  mini_pawo_flat_env_cfg,
)

from dataclasses import replace

from .recipes import (
  ACTUATOR_LATENCY, BASELINE, BAM, BAM_ACTUATOR_LATENCY, BAM_SENSOR_LATENCY,
  BAM_SENSOR_ACTUATOR_LATENCY, SENSOR_LATENCY,
)
from .rl_cfg import mini_pawo_ppo_runner_cfg 

for task_id, recipe, experiment_name in (
  ("MiniPawo-Flat", BASELINE, "MiniPawo_velocity"),
  ("MiniPawo-Flat-SensorLatency", SENSOR_LATENCY, "MiniPawo_sensor_latency"),
  ("MiniPawo-Flat-ActuatorLatency", ACTUATOR_LATENCY, "MiniPawo_actuator_latency"),
  ("MiniPawo-Flat-BAM", BAM, "MiniPawo_bam"),
  ("MiniPawo-Flat-BAM-ActuatorLatency", BAM_ACTUATOR_LATENCY, "MiniPawo_bam_actuator_latency"),
  ("MiniPawo-Flat-BAM-SensorLatency-ActuatorLatency", BAM_SENSOR_ACTUATOR_LATENCY, "MiniPawo_bam_sensor_actuator_latency"),
  ("MiniPawo-Flat-BAM-SensorLatency", BAM_SENSOR_LATENCY, "MiniPawo_bam_sensor_latency"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=mini_pawo_flat_env_cfg(recipe=recipe),
    play_env_cfg=mini_pawo_flat_env_cfg(play=True, recipe=recipe),
    rl_cfg=replace(mini_pawo_ppo_runner_cfg(), experiment_name=experiment_name),
    runner_cls=VelocityOnPolicyRunner,
  )
