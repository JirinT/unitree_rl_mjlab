from mjlab.tasks.registry import register_mjlab_task
from src.tasks.hand_reach_target.rl import HandReachOnPolicyRunner

from .env_cfgs import unitree_h1_2_left_arm_reach_env_cfg
from .rl_cfg import unitree_h1_2_left_arm_reach_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H1_2-LeftArm-Reach",
  env_cfg=unitree_h1_2_left_arm_reach_env_cfg(),
  play_env_cfg=unitree_h1_2_left_arm_reach_env_cfg(play=True),
  rl_cfg=unitree_h1_2_left_arm_reach_ppo_runner_cfg(),
  runner_cls=HandReachOnPolicyRunner,
)
