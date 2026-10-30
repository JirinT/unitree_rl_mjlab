"""Position-command latency, independent of sensor latency and motor physics."""

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg


def add_actuator_latency(cfg: ManagerBasedRlEnvCfg) -> None:
  """Delay position commands by 3-6 physics steps (15-30 ms at dt=0.005).

  Apply after the actuator-selection modifier so it delays the final actuator.
  Keep this modifier active in both training and playback.
  """
  robot = cfg.scene.entities["robot"]
  actuators = []
  for actuator in robot.articulation.actuators:
    if "delay_max_lag" in {field.name for field in fields(actuator)}:
      actuator = replace(actuator, delay_min_lag=3, delay_max_lag=6)
    else:
      # Older mjlab versions implement latency using a separate wrapper.
      from mjlab.actuator import DelayedActuatorCfg
      actuator = DelayedActuatorCfg(
        base_cfg=actuator,
        delay_target="position",
        delay_min_lag=3,
        delay_max_lag=6,
      )
    actuators.append(actuator)
  robot.articulation = replace(robot.articulation, actuators=tuple(actuators))
