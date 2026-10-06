"""Mini Pawo humanoid 10 DOF constants."""

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

MINI_PAWO_XML: Path = (
  SRC_PATH / "assets" / "robots" / "mini_pawo" / "xmls" / "robot.xml"  # Adjust filename if necessary
)
assert MINI_PAWO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, MINI_PAWO_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MINI_PAWO_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

MINI_PAWO_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*hip_yaw",
        ".*hip_roll",
        ".*hip_pitch",
        ".*knee",
        ".*ankle",
    ),
    stiffness=10.0,
    damping=2.5,
    effort_limit=2.94,
    armature=0.015,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0., 0., 0.01),   # Adjusted starting height so feet clear the ground
  joint_pos={
    ".*hip_pitch": 0.0,
    ".*knee": -0.523598776,
    ".*ankle": 0.523598776,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##
# Onshape-to-MuJoCo usually applies class="collision" to geoms rather than 
# renaming them with a _collision suffix. We target the foot mesh geoms directly.
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
MINI_PAWO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MINI_PAWO_ACTUATOR_LEG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_mini_pawo_robot_cfg() -> EntityCfg:
  """Get a fresh Mini Pawo humanoid robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=MINI_PAWO_ARTICULATION,
  )


MINI_PAWO_ACTION_RANGE_RAD = 1.0  # 57 deg
MINI_PAWO_ACTION_SCALE: dict[str, float] = {}
for a in MINI_PAWO_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  names = a.target_names_expr
  for n in names:
    MINI_PAWO_ACTION_SCALE[n] = MINI_PAWO_ACTION_RANGE_RAD


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_mini_pawo_robot_cfg())
  model = robot.spec.compile()
  
  model.opt.gravity[:] = 0.0  # Uncomment to test without gravity collapsing the legs

  data = mujoco.MjData(model)
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key_id)

  viewer.launch(model, data)