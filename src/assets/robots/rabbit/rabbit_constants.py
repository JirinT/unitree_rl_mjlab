"""Rabbit humanoid 14 dof constants."""

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

RABBIT_XML: Path = (
  SRC_PATH / "assets" / "robots" / "rabbit" / "xmls" / "robot.xml"
)
assert RABBIT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, RABBIT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(RABBIT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

RABBIT_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*HipYaw",
    ".*HipRoll",
    ".*HipPitch",
    ".*Knee",
    ".*Ankle",
  ),
  stiffness=20.0,
  damping=1.28,
  effort_limit=10.0,
  armature=0.005,
)
RABBIT_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*ShoulderPitch",
    ".*Elbow",
  ),
  stiffness=20.0,
  damping=1.28,
  effort_limit=10.0,
  armature=0.005,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0., 0., 0.9),
  joint_pos={
    ".*HipPitch": 0.0,
    ".*Knee": 0.0,
    ".*Ankle": 0.0,
    ".*ShoulderPitch": 0.0,
    ".*Elbow": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_collision$",),
  contype=1.0,
  conaffinity=1.,
  condim=3.,
  priority=1.,
  friction=(0.6,),
)


##
# Final config.
##
RABBIT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(RABBIT_ACTUATOR_LEG, RABBIT_ACTUATOR_ARM),
  soft_joint_pos_limit_factor=0.9,
)


def get_rabbit_robot_cfg() -> EntityCfg:
  """Get a fresh Rabbit humanoid robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=RABBIT_ARTICULATION,
  )


RABBIT_ACTION_RANGE_RAD = 1.0  # 57 deg
RABBIT_ACTION_SCALE: dict[str, float] = {}
for a in RABBIT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  names = a.target_names_expr
  for n in names:
    RABBIT_ACTION_SCALE[n] = RABBIT_ACTION_RANGE_RAD


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_rabbit_robot_cfg())
  model = robot.spec.compile()
  model.opt.gravity[:] = 0.0

  data = mujoco.MjData(model)
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key_id)

  viewer.launch(model, data)