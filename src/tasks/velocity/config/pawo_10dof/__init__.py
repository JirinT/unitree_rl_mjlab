from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pawo_flat_env_cfg,
)
from .rl_cfg import pawo_ppo_runner_cfg

register_mjlab_task(
  task_id="PAWO-Flat",
  env_cfg=pawo_flat_env_cfg(),
  play_env_cfg=pawo_flat_env_cfg(play=True),
  rl_cfg=pawo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
