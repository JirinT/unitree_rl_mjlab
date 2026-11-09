"""Use the supplied Unitree MuJoCo deployment plant for H1_2 training."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.unitree_h1_2.deployment import get_deployment_robot_cfg


def match_unitree_mujoco(cfg: ManagerBasedRlEnvCfg, *, free_base: bool = False) -> None:
  """Nominal deployment model; asynchronous DDS timing is not simulated.

  free_base=False matches the Linux XML's inherited floating-base defaults.
  Use True only after overriding damping/frictionloss/armature to zero there too.
  """
  cfg.scene.entities["robot"] = get_deployment_robot_cfg(free_base=free_base)
  # No legacy capsule-foot names or nominal physical randomization. Start with
  # measured nominal parity; optional uncertainty can be added as later modifiers.
  for name in ("foot_friction", "base_com", "encoder_bias"):
    cfg.events.pop(name, None)
  cfg.sim = deepcopy(cfg.sim)
  cfg.sim.mujoco.timestep = 0.005
  cfg.sim.mujoco.integrator = "euler"
  # Match MuJoCo XML defaults used by the supplied deployment scene.
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.tolerance = 1e-8
  cfg.sim.mujoco.ls_tolerance = 0.01
  cfg.sim.mujoco.impratio = 1.0
  cfg.sim.mujoco.cone = "pyramidal"
  cfg.sim.mujoco.jacobian = "auto"
  cfg.sim.mujoco.solver = "newton"
  cfg.sim.mujoco.ccd_iterations = 35
  cfg.decimation = 4
