"""Unitree H1_2 tray loco-manipulation environment configurations."""

from src.assets.robots.unitree_h1_2_tray.h1_2_tray_constants import (
  H1_2_TRAY_ACTION_SCALE, get_h1_2_tray_robot_cfg, get_tray_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.loco_manipulation.loco_manipulation_env_cfg import make_locomanipulation_env_cfg
import mujoco

import torch
import math
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_mul, quat_from_euler_xyz



def reset_tray_to_hands(env, env_ids,
                        robot_cfg=SceneEntityCfg("robot"),
                        tray_cfg=SceneEntityCfg("tray"),
                        left_site="left_palm", right_site="right_palm"):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  env.scene.write_data_to_sim()
  env.sim.forward()

  robot = env.scene[robot_cfg.name]
  tray  = env.scene[tray_cfg.name]
  lid = robot.find_sites(left_site)[0][0]
  rid = robot.find_sites(right_site)[0][0]

  pL = robot.data.site_pos_w[env_ids, lid]
  pR = robot.data.site_pos_w[env_ids, rid]
  mid = 0.5 * (pL + pR)
  quat = robot.data.site_quat_w[env_ids, lid]

  # Correct the constant 90deg offset between the palm frame and the tray frame.
  # Applied on the right -> rotation about the tray's own z (yaw). Flip the sign
  # of the angle if it turns the wrong way.
  n = len(env_ids)
  zero = torch.zeros(n, device=env.device)
  yaw_fix = quat_from_euler_xyz(zero, zero, zero - math.pi / 2)   # (n, 4)
  quat = quat_mul(quat, yaw_fix)

  root_state = torch.zeros((n, 13), device=env.device)
  root_state[:, 0:3] = mid
  root_state[:, 3:7] = quat
  tray.write_root_state_to_sim(root_state, env_ids)

def weld_tray_to_hands(spec):
  """
  Weld the trays two grip sites to the palm sites.
  """
  for palm_site, tray_site in (
      ("robot/left_palm", "tray/tray_grip_L"),
      ("robot/right_palm", "tray/tray_grip_R"),
  ):
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_SITE
    eq.name1 = palm_site
    eq.name2 = tray_site
    eq.solref = [0.02, 1]
    eq.solimp = [0.9, 0.95, 0.001, 0.5, 2]

def unitree_h1_2_tray_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1_2 rough-terrain tray loco-manipulation configuration."""
  cfg = make_locomanipulation_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {
    "robot": get_h1_2_tray_robot_cfg(),
    "tray": get_tray_cfg(),
  }
  cfg.scene.spec_fn = weld_tray_to_hands

  # Set raycast sensor frame to H1_2 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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
  joint_pos_action.scale = H1_2_TRAY_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.55

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["reset_tray_to_hands"] = EventTermCfg(
    func=reset_tray_to_hands,
    mode="reset",
    params={
        "robot_cfg": SceneEntityCfg("robot"),
        "tray_cfg": SceneEntityCfg("tray"),
    },
)
  
  cfg.rewards["pose"].params["std_standing"] = {
    # Lower body + waist (same as velocity task).
    r".*hip_yaw.*": 0.05,
    r".*hip_pitch.*": 0.05,
    r".*hip_roll.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle_pitch.*": 0.05,
    r".*ankle_roll.*": 0.05,
    r".*torso.*": 0.05,
    # Arms (loosened so they can hold + re-level the tray). PLACEHOLDER.
    r".*shoulder.*": 0.3,
    r".*elbow.*": 0.3,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_yaw.*": 0.15,
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*torso.*": 0.15,
    # Arms. NOTE: these are tighter (0.1) than std_standing's arm values - during
    # walking the arms will be pulled toward 0 harder, which may fight tray leveling
    # while moving. Consider loosening to ~0.3 if the tray tips during locomotion.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_yaw.*": 0.25,
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*torso.*": 0.25,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
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


def unitree_h1_2_tray_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1_2 flat-terrain tray loco-manipulation configuration.
  """
  cfg = unitree_h1_2_tray_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
