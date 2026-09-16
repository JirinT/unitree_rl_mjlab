"""PAWO 6 dof constants."""

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

PAWO_6DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "pawo_6dof" / "xmls" / "robot.xml"
)
assert PAWO_6DOF_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PAWO_6DOF_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PAWO_6DOF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

PAWO_ACTUATOR_D845WP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_knee.*",
    ".*_touch",
  ),
  stiffness=50.0,
  damping=3.2,
  effort_limit=10.0,
  armature=6e-4,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.02),
  joint_pos={
    ".*_hip_pitch_joint": 0.,
    ".*_knee_joint": 0.,
    ".*_touch": 0.,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_collision$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

PAWO_6DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PAWO_ACTUATOR_D845WP,),
  soft_joint_pos_limit_factor=0.9,
)


def get_pawo_6dof_robot_cfg() -> EntityCfg:
  """Get a fresh PAWO 6dof robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PAWO_6DOF_ARTICULATION,
  )


PAWO_6DOF_ACTION_RANGE_RAD = 1.0  # 57 deg per joint; retune once a gait emerges.
PAWO_6DOF_ACTION_SCALE: dict[str, float] = {}
for a in PAWO_6DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  names = a.target_names_expr
  for n in names:
    PAWO_6DOF_ACTION_SCALE[n] = PAWO_6DOF_ACTION_RANGE_RAD


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pawo_6dof_robot_cfg())
  model = robot.spec.compile()
  model.opt.gravity[:] = 0.0

  data = mujoco.MjData(model)
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key_id)

  viewer.launch(model, data)
