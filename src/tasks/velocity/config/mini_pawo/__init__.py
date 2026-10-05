from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  mini_pawo_flat_env_cfg,
)
# Assuming you renamed your PPO config to match the new robot. 
# If not, you can leave this as rabbit_ppo_runner_cfg.
from .rl_cfg import mini_pawo_ppo_runner_cfg 

register_mjlab_task(
  task_id="MiniPawo-Flat",
  env_cfg=mini_pawo_flat_env_cfg(),
  play_env_cfg=mini_pawo_flat_env_cfg(play=True),
  rl_cfg=mini_pawo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)