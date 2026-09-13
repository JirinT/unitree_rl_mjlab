from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  rabbit_flat_env_cfg,
)
from .rl_cfg import rabbit_ppo_runner_cfg

register_mjlab_task(
  task_id="Rabbit-Flat",
  env_cfg=rabbit_flat_env_cfg(),
  play_env_cfg=rabbit_flat_env_cfg(play=True),
  rl_cfg=rabbit_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
