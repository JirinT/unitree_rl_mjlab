"""Unitree H1_2 constants: left arm only.
"""

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
  SRC_PATH / "assets" / "robots" / "unitree_h1_2" / "xmls" / "h1_2.xml"
)
assert H1_2_XML.exists()

LEFT_ARM_JOINTS: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
)

# End-effector site already defined in the XML on left_wrist_yaw_link.
EE_SITE_NAME = "left_palm"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_2_XML))
  spec.assets = get_assets(spec.meshdir)

  # Delete every joint except the left-arm ones.
  for joint in list(spec.joints):
    if joint.name not in LEFT_ARM_JOINTS:
      spec.delete(joint)
  return spec

##
# Actuator config (same gains as the full H1_2 config).
##

H1_2_ACTUATOR_GO2HV_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
  ),
  stiffness=19.7,
  damping=1.3,
  effort_limit=40.0,
  armature=0.005,
)
H1_2_ACTUATOR_GO2HV_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
  ),
  stiffness=7.9,
  damping=0.5,
  effort_limit=18.0,
  armature=0.002,
)


##
# Keyframe config.
##

# Base is fixed, so pos is the (pinned) pelvis position.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0),
  joint_pos={
    "left_shoulder_pitch_joint": 0.28,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.52,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Arm vs. frozen body (torso, pelvis, head, legs) and arm vs. itself.
ARM_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim=1,
)


##
# Final config.
##

H1_2_LEFT_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_2_ACTUATOR_GO2HV_1,
    H1_2_ACTUATOR_GO2HV_2,
  ),
  soft_joint_pos_limit_factor=1.0,  # use the full real range
)


def get_h1_2_left_arm_robot_cfg() -> EntityCfg:
  """Get a fresh left-arm-only H1_2 configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(ARM_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_2_LEFT_ARM_ARTICULATION,
  )

H1_2_LEFT_ARM_ACTION_SCALE: dict[str, float] = {}
for a in H1_2_LEFT_ARM_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    H1_2_LEFT_ARM_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_2_left_arm_robot_cfg())
  model = robot.spec.compile()

  # Slider range = real joint range, start at home and hold it.
  data = mujoco.MjData(model)
  for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    model.actuator_ctrllimited[i] = 1
    model.actuator_ctrlrange[i] = model.jnt_range[jid]
    short = model.joint(jid).name.split("/")[-1]
    data.qpos[model.jnt_qposadr[jid]] = HOME_KEYFRAME.joint_pos[short]
  mujoco.mj_forward(model, data)
  for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

  viewer.launch(model, data)