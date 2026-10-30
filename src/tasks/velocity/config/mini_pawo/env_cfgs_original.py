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
# Keep aligned with num_steps_per_env in mini_pawo/rl_cfg.py.
NUM_STEPS_PER_ENV = 24

def mini_pawo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini Pawo humanoid rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_mini_pawo_robot_cfg()}
  # BAM requires these fields expanded before any simulation steps, including
  # callers that construct the task directly without train.py or play.py.
  from src.assets.robots.mini_pawo.actuator_cfg import describe_actuators
  if describe_actuators(cfg.scene.entities["robot"])["actuator_model"] == "bam":
    from src.vendor.bam.mjlab import bam_init
    cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "torso"
  
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

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MINI_PAWO_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.debug_vis = False
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.viz.y_offset = -0.5
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names
  cfg.observations["actor"].terms["phase"].params["period"] = GAIT_PERIOD

  # MicroDuck-style sensor latency (control steps, not physics substeps).
  # Clone terms because the base config shares them with the privileged critic.
  # At 50 Hz: IMU 0/20 ms, joint velocity 20 ms. Keep these in playback too.
  for name in ("base_ang_vel", "projected_gravity", "joint_vel"):
    term = deepcopy(cfg.observations["actor"].terms[name])
    term.delay_min_lag = 1 if name == "joint_vel" else 0
    term.delay_max_lag = 1
    term.delay_update_period = 0 if name == "joint_vel" else 64
    cfg.observations["actor"].terms[name] = term

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.rewards["pose"].params["walking_threshold"] = 0.01 # is 0.1 in velocity_env_cfg.py
  cfg.rewards["pose"].params["running_threshold"] = 1 # was 0.7
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_yaw": 0.15, # was 0.15
    r".*hip_pitch": 0.5, # was 0.5
    r".*hip_roll": 0.15, # was 0.15
    r".*knee": 0.5, # was 0.5
    r".*ankle": 0.15, # was 0.15
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_yaw": 0.25,
    r".*hip_pitch": 0.5,
    r".*hip_roll": 0.25,
    r".*knee": 0.5,
    r".*ankle": 0.15,
  }

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

  cfg.terminations["torso_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": -0.1}, # Adjusted for Pawo's shorter stature
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    )

  cfg.events["push_robot"].interval_range_s=(3., 6.)
  cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.4, 0.4),
      "y": (-0.4, 0.4),
      "z": (0., 0.),
      "roll": (0., 0.),
      "pitch": (0., 0.),
      "yaw": (0., 0.),
  }

  cfg.events["base_com"].params["ranges"] = {
          0: (-0.005, 0.005),
          1: (-0.005, 0.005),
          2: (-0.01, 0.01),
        }

  cfg.events["reset_robot_joints"].params["position_range"]= (-0.05, 0.05)

  cfg.rewards["action_rate_l2"].weight = -0.1
  # Start with walking, then increase smoothness and standing practice.
  # twist_cmd.rel_standing_envs = 0.02
  # cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
  #   func=velocity_mdp.reward_weight,
  #   params={
  #     "reward_name": "action_rate_l2",
  #     "weight_stages": [
  #       {"step": 0, "weight": -0.1},
  #       {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
  #       {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
  #       {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
  #       {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
  #       {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
  #     ],
  #   },
  # )
  # cfg.curriculum["standing_envs"] = CurriculumTermCfg(
  #   func=velocity_mdp.standing_envs_curriculum,
  #   params={
  #     "command_name": "twist",
  #     "standing_stages": [
  #       {"step": 0, "rel_standing_envs": 0.02},
  #       {"step": 500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.05},
  #       {"step": 750 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.1},
  #       {"step": 1000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.15},
  #       {"step": 1500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.2},
  #       {"step": 2000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.25},
  #     ],
  #   },
  # )

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