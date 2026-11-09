"""Optional Unitree MuJoCo plant; the original H1_2 asset is unchanged.

The frozen XML supplies mesh contacts/inertias. Runtime construction replaces its
SDK motors with mjlab explicit PD motors, preserving natural joint/action order.
"""

from copy import deepcopy
from functools import partial
from pathlib import Path
import re

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .h1_2_constants import (
  H1_2_ARTICULATION, HOME_KEYFRAME, get_assets, get_spec as get_baseline_spec,
)


DEPLOYMENT_XML = Path(__file__).parent / "xmls/h1_2_deployment.xml"
DEPLOYMENT_PROFILE = "unitree_mujoco_h1_2_v1"


def get_deployment_spec(*, free_base: bool = False) -> mujoco.MjSpec:
  """Match the supplied Linux model, optionally removing root drag in BOTH sims."""
  spec = mujoco.MjSpec.from_file(str(DEPLOYMENT_XML))
  spec.assets = get_assets(spec.meshdir)
  for sensor in tuple(spec.sensors):
    spec.delete(sensor)
  for motor in tuple(spec.actuators):
    spec.delete(motor)
  # Name collision geoms for configuration/diagnostics without changing masks,
  # contact dimensions, friction, priority, shape or local transform.
  for body in spec.bodies:
    for i, geom in enumerate(body.geoms):
      if not geom.name:
        kind = "collision" if geom.contype or geom.conaffinity else "visual"
        geom.name = f"{body.name}_{kind}_{i}"
  root = spec.joint("floating_base_joint")
  if free_base:
    # MuJoCo versions expose damping as a scalar or a coefficient vector.
    if isinstance(root.damping, float):
      root.damping = 0.0
    else:
      root.damping[:] = 0.0
    root.frictionloss = 0.0
    root.armature = 0.0
  # Keep deployment armatures already supplied by the user. Passive hinge
  # damping/friction remain 1 / .2, inherited from the deployment XML.
  for group in H1_2_ARTICULATION.actuators:
    for joint in spec.joints:
      if any(re.fullmatch(pattern, joint.name) for pattern in group.target_names_expr):
        joint.armature = group.armature
  baseline = get_baseline_spec()
  for side in ("left", "right"):
    for site_name, body_name in ((f"{side}_foot", f"{side}_ankle_roll_link"),
                                 (f"{side}_palm", f"{side}_wrist_yaw_link")):
      site = baseline.site(site_name)
      spec.body(body_name).add_site(
        name=site_name, pos=site.pos, quat=site.quat, size=site.size,
      )
  for name, kind in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
                      ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
                      ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
    spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                  objtype=mujoco.mjtObj.mjOBJ_BODY, objname="torso_link")
  return spec


def get_deployment_robot_cfg(*, free_base: bool = False) -> EntityCfg:
  grouped = {}
  # Split groups where the deployment torque limits differ, preserving gains
  # and armature. Action scaling is intentionally NOT recalculated from limits.
  for group in H1_2_ARTICULATION.actuators:
    for pattern in group.target_names_expr:
      limit = 60.0 if "ankle_pitch" in pattern else (19.0 if "wrist" in pattern else group.effort_limit)
      key = (group.stiffness, group.damping, limit, group.armature)
      grouped.setdefault(key, []).append(pattern)
  groups = tuple(
    IdealPdActuatorCfg(
      target_names_expr=tuple(patterns), stiffness=kp, damping=kd,
      effort_limit=limit, armature=armature,
      frictionloss=0.2,  # Passive damping=1 remains in the XML.
    )
    for (kp, kd, limit, armature), patterns in grouped.items()
  )
  return EntityCfg(
    init_state=deepcopy(HOME_KEYFRAME),
    spec_fn=partial(get_deployment_spec, free_base=free_base),
    sort_actuators=True,
    collisions=(),  # Preserve the XML's actual contacts; no capsule overrides.
    articulation=EntityArticulationInfoCfg(actuators=groups, soft_joint_pos_limit_factor=0.9),
  )


def deployment_metadata(env) -> dict:
  """Explicit PD gains live in the controller, not actuator_gainprm (which is 1)."""
  robot = env.scene["robot"]
  spec_fn = robot.cfg.spec_fn
  if getattr(spec_fn, "func", spec_fn) is not get_deployment_spec:
    return {}
  gains = {}
  for actuator in robot.actuators:
    if not isinstance(actuator.cfg, IdealPdActuatorCfg):
      raise ValueError("Deployment profile expects explicit PD actuators.")
    for name in actuator.target_names:
      if name in gains:
        raise ValueError(f"Duplicate deployment actuator: {name}")
      gains[name] = (actuator.cfg.stiffness, actuator.cfg.damping, actuator.cfg.effort_limit)
  ordered = [gains[name] for name in robot.joint_names]
  model = env.sim.mj_model
  ids = [model.joint(f"robot/{name}").id for name in robot.joint_names]
  dofs = model.jnt_dofadr[ids]
  root = model.joint("robot/floating_base_joint").dofadr[0]
  return {
    "joint_stiffness": [v[0] for v in ordered],
    "joint_damping": [v[1] for v in ordered],
    "joint_effort_limits": [v[2] for v in ordered],
    "deployment_profile": DEPLOYMENT_PROFILE,
    "actuator_model": "explicit_pd",
    "collision_profile": "unitree_h1_2_mesh",
    "physics_timestep": env.cfg.sim.mujoco.timestep,
    "policy_timestep": env.step_dt,
    "physics_integrator": env.cfg.sim.mujoco.integrator,
    "joint_armature": model.dof_armature[dofs].tolist(),
    "joint_passive_damping": model.dof_damping[dofs].tolist(),
    "joint_frictionloss": model.dof_frictionloss[dofs].tolist(),
    "floating_base_armature": model.dof_armature[root:root + 6].tolist(),
    "floating_base_damping": model.dof_damping[root:root + 6].tolist(),
    "floating_base_frictionloss": model.dof_frictionloss[root:root + 6].tolist(),
  }
