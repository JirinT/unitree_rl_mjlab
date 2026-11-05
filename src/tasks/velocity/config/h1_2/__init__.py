from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_h1_2_flat_env_cfg,
  unitree_h1_2_rough_env_cfg,
)
from .recipes import ARM_INTERVENTION, BASELINE, HAND_TRACKING
from .rl_cfg import unitree_h1_2_ppo_runner_cfg

for task_id, env_cfg_factory, recipe, experiment_name in (
  ("Unitree-H1_2-Rough", unitree_h1_2_rough_env_cfg, BASELINE, "h1_2_velocity"),
  ("Unitree-H1_2-Flat", unitree_h1_2_flat_env_cfg, BASELINE, "h1_2_velocity"),
  ("Unitree-H1_2-Rough-HandTracking", unitree_h1_2_rough_env_cfg, HAND_TRACKING, "h1_2_hand_tracking"),
  ("Unitree-H1_2-Flat-HandTracking", unitree_h1_2_flat_env_cfg, HAND_TRACKING, "h1_2_hand_tracking"),
  ("Unitree-H1_2-Rough-ArmIntervention", unitree_h1_2_rough_env_cfg, ARM_INTERVENTION, "h1_2_arm_intervention"),
  ("Unitree-H1_2-Flat-ArmIntervention", unitree_h1_2_flat_env_cfg, ARM_INTERVENTION, "h1_2_arm_intervention"),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg_factory(recipe=recipe),
    play_env_cfg=env_cfg_factory(play=True, recipe=recipe),
    rl_cfg=replace(unitree_h1_2_ppo_runner_cfg(), experiment_name=experiment_name),
    runner_cls=VelocityOnPolicyRunner,
  )
