"""Mini Pawo humanoid 10-DOF velocity environment configurations."""

from src.assets.robots.mini_pawo.minipawo_constants import (
  MINI_PAWO_ACTION_SCALE,
  get_mini_pawo_robot_cfg,
)
import math
from copy import deepcopy
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity import mdp as velocity_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

GAIT_PERIOD = 0.4

def mini_pawo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini Pawo humanoid flat velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500 # Allow up to 500 iterations when calculating collisions between convex shapes
  cfg.sim.contact_sensor_maxmatch = 500 # Reserve space for up to 500 matching contacts per contact sensor, per environment, before sensor reduction/aggregation
  cfg.sim.nconmax = 48 # Allocate physical contact storage averaging 48 contacts per environment

  cfg.scene.entities = {"robot": get_mini_pawo_robot_cfg()}
  
  site_names = ("left_foot_site", "right_foot_site")
  geom_names = ("left_foot_collision", "right_foot_collision")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      # Target the specific foot bodies
      pattern=r"^(left_foot|right_foot)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MINI_PAWO_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.debug_vis = False
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.viz.y_offset = -0.5
  # this is under curriculum, which overwrites it!!! These values need to start same as the curriculum
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  cfg.observations["actor"].terms["phase"].params["period"] = GAIT_PERIOD

  pose = cfg.rewards["pose"].params
  pose["walking_threshold"] = 0.01
  pose["running_threshold"] = float("inf") # disable running mode
  pose["std_standing"] = {".*": 0.05}
  pose["std_walking"] = {
    r".*hip_yaw": 0.2,
    r".*hip_pitch": 0.5,
    r".*hip_roll": 0.2,
    r".*knee": 0.5,
    r".*ankle": 0.2, 
  }
  pose["std_running"] = deepcopy(pose["std_walking"])

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["target_height"] = 0.03
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_gait"].params["period"] = GAIT_PERIOD
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  cfg.terminations["fell_over"] = TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
  cfg.events["push_robot"].interval_range_s=(3., 6.)
  cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.4, 0.4),
      "y": (-0.4, 0.4),
      "z": (0., 0.),
      "roll": (0., 0.),
      "pitch": (0., 0.),
      "yaw": (0., 0.),
  }

  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  cfg.events["base_com"].params["ranges"] = {
          0: (-0.015, 0.015),
          1: (-0.015, 0.015),
          2: (-0.015, 0.015),
        }

  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None) # keep the command constant during training

  if play:
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0., 0.),
          "y": (-0., 0.),
          "z": (-0., 0.),
          "roll": (-0., 0.),
          "pitch": (-0., 0.),
          "yaw": (-0., 0.),
        },
      },
    )

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def mini_pawo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini Pawo humanoid flat terrain velocity configuration."""
  cfg = mini_pawo_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # twist_cmd.ranges.lin_vel_x = (0., 0.0)
    # twist_cmd.ranges.lin_vel_y = (0., 0.0)
    # twist_cmd.ranges.ang_vel_z = (0., 0.0)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg