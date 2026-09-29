"""Reaching task configuration.

Factory for a fixed-base arm reaching a random 3D target with an end-effector
site. Robot-specific configs call the factory and fill in the "Set per-robot"
fields (see reach_h1_2_left_arm_env_cfg.py).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.hand_reach_target import mdp
from src.tasks.hand_reach_target.mdp import ReachCommandCfg


def make_reach_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base reaching task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_to_target": ObservationTermCfg(
      func=mdp.ee_to_target,
      params={
        "command_name": "reach_target",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "target_pos": ObservationTermCfg(
      func=mdp.target_pos_rel_origin,
      params={"command_name": "reach_target"},
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # Fixed base, no privileged state needed: critic sees the clean actor obs.
  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "reach_target": ReachCommandCfg(
      resampling_time_range=(3.0, 5.0),
      debug_vis=True,
      # Target box in the env-origin frame. Override per-robot.
      target_position_range=ReachCommandCfg.TargetPositionRangeCfg(
        x=(0.3, 0.6),
        y=(-0.3, 0.3),
        z=(0.2, 0.6),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    # Positions the (fixed) base at env_origins.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Rewards
  ##

  ee_asset_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

  rewards = {
    # Coarse kernel: gradient far from the target.
    "reach_coarse": RewardTermCfg(
      func=mdp.ee_position_kernel,
      weight=1.0,
      params={
        "command_name": "reach_target",
        "std": 0.25,
        "asset_cfg": ee_asset_cfg,
      },
    ),
    # Fine kernel: precision near the target.
    "reach_fine": RewardTermCfg(
      func=mdp.ee_position_kernel,
      weight=1.0,
      params={
        "command_name": "reach_target",
        "std": 0.05,
        "asset_cfg": ee_asset_cfg,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 1.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=55,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
