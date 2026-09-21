"""Unitree H1_2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_h1_2_tray" / "xmls" / "h1_2.xml"
)
TRAY_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_h1_2_tray" / "xmls" / "tray.xml"
)
assert H1_2_XML.exists()
assert TRAY_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

def get_tray_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(TRAY_XML))


##
# Actuator config.
##

H1_2_ACTUATOR_M107_24_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw.*",
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    "torso_joint",
  ),
  stiffness=98.7,
  damping=6.3,
  effort_limit=200.0,
  armature=0.025,
)
H1_2_ACTUATOR_M107_24_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_knee.*",
  ),
  stiffness=157.7,
  damping=10.1,
  effort_limit=300.0,
  armature=0.04,
)
H1_2_ACTUATOR_GO2HV_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
  ),
  stiffness=19.7,
  damping=1.3,
  effort_limit=40.0,
  armature=0.005,
)
H1_2_ACTUATOR_GO2HV_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_pitch.*",
    ".*_wrist_roll.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=7.9,
  damping=0.5,
  effort_limit=18.0,
  armature=0.002,
)



##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.02),
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.3,
    ".*_shoulder_pitch_joint": 0.28,
    ".*_elbow_joint": 0.52,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H1_2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_2_ACTUATOR_M107_24_2,
    H1_2_ACTUATOR_M107_24_1,
    H1_2_ACTUATOR_GO2HV_1,
    H1_2_ACTUATOR_GO2HV_2,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_2_robot_cfg() -> EntityCfg:
  """Get a fresh H1_2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_2_ARTICULATION,
  )

def get_tray_cfg() -> EntityCfg:
  """
  Get a fresh TRAY configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.5, 0, 0.05)),
    spec_fn=get_tray_spec,
  )

H1_2_ACTION_SCALE: dict[str, float] = {}
for a in H1_2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H1_2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_2_robot_cfg())
  spec = robot.spec

  # Freeze every robot joint (base included) at its zero/nominal
  # position
  for a in list(spec.actuators):
    spec.delete(a)
  for j in list(spec.joints):
    spec.delete(j)
  for k in list(spec.keys):
    spec.delete(k)

  tray_spec = mujoco.MjSpec.from_file(str(TRAY_XML))
  tray_body = tray_spec.body("tray")

  frame = spec.worldbody.add_frame(
      pos=[0.32, 0, 0.175],
      quat=[0.7071068, 0, 0, -0.7071068],
  )
  frame.attach_body(tray_body, "", "")

  for palm_site, tray_site in (
      ("left_palm", "tray_grip_L"),
      ("right_palm", "tray_grip_R"),
  ):
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_SITE
    eq.name1 = palm_site
    eq.name2 = tray_site
    eq.solref = [0.02, 1]
    eq.solimp = [0.9, 0.95, 0.001, 0.5, 2]

  model = spec.compile()
  model.opt.gravity[:] = 0
  data = mujoco.MjData(model)
  viewer.launch(model, data)