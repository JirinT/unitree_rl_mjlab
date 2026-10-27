"""Mini Pawo humanoid 10 DOF constants."""

from dataclasses import fields, replace
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

MINI_PAWO_ACTUATOR_MODEL = "bam"  # "builtin" or "bam"

MINI_PAWO_BAM_JSON_PATH = SRC_PATH / "vendor" / "bam" / "params" / "waveshare_st3215" / "m6.json"
MINI_PAWO_BAM_VIN: float | None = None  # None: controller/model default
MINI_PAWO_BAM_KP_FW: float | None = None  # Firmware gain, not PD stiffness
MINI_PAWO_BAM_DELAY_MIN = 3
MINI_PAWO_BAM_DELAY_MAX = 6

MINI_PAWO_LEG_JOINTS = (
    ".*hip_yaw",
    ".*hip_roll",
    ".*hip_pitch",
    ".*knee",
    ".*ankle",
)

MINI_PAWO_BUILTIN_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
    target_names_expr=MINI_PAWO_LEG_JOINTS,
    stiffness=10.0,
    damping=2.5,
    effort_limit=2.94,
    armature=0.015,
)


def make_mini_pawo_bam_actuator(
    *, json_path=None, motor_name=None, model=None, vin=None, kp_fw=None,
    delay_min_lag=0, delay_max_lag=0,
):
  """
  Construct BAM using the documented API
  Parameters:
    json_path: (str) absolute path to the identified json parameters
  """
  from src.vendor.bam.mjlab import BamActuatorCfg

  actuator = BamActuatorCfg(
      json_path=json_path, motor_name=motor_name, model=model,
      target_names_expr=MINI_PAWO_LEG_JOINTS, vin=vin, kp_fw=kp_fw)
  
  if delay_max_lag:
    if "delay_max_lag" in {field.name for field in fields(actuator)}:
      # for BAM:
      actuator = replace(actuator, delay_min_lag=delay_min_lag,
                         delay_max_lag=delay_max_lag)
    else:
      # For mjlab:
      from mjlab.actuator import DelayedActuatorCfg
      actuator = DelayedActuatorCfg(
          base_cfg=actuator,
          delay_target="position",
          delay_min_lag=delay_min_lag,
          delay_max_lag=delay_max_lag,
      )
      
  return actuator


def _default_leg_actuator():
  if MINI_PAWO_ACTUATOR_MODEL == "builtin":
    return MINI_PAWO_BUILTIN_ACTUATOR_LEG

  return make_mini_pawo_bam_actuator(
      json_path= MINI_PAWO_BAM_JSON_PATH, 
      vin=MINI_PAWO_BAM_VIN, 
      kp_fw=MINI_PAWO_BAM_KP_FW,
      delay_min_lag=MINI_PAWO_BAM_DELAY_MIN,
      delay_max_lag=MINI_PAWO_BAM_DELAY_MAX)


mini_pawo_actuator_leg_cfg = _default_leg_actuator()

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0., 0., 0.01),   # Adjusted starting height so feet clear the ground
  joint_pos={
    "right_hip_roll": 0.0872664626,
    "left_hip_roll": -0.0872664626,
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
  actuators=(mini_pawo_actuator_leg_cfg,),
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


MINI_PAWO_ACTION_RANGE_RAD = 1.0  # ~ 57 deg, Can be adjusted in the modifiers.
MINI_PAWO_ACTION_SCALE: dict[str, float] = {}
for name in MINI_PAWO_LEG_JOINTS:
  MINI_PAWO_ACTION_SCALE[name] = MINI_PAWO_ACTION_RANGE_RAD


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_mini_pawo_robot_cfg())
  model = robot.spec.compile()

  total_mass = mujoco.mj_getTotalmass(model)
  print(f"Total robot mass: {total_mass:.3f} kg")

  model.opt.gravity[:] = 0.0  # Uncomment to test without gravity collapsing the legs

  data = mujoco.MjData(model)
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key_id)

  viewer.launch(model, data)