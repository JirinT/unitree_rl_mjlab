"""Sensor delays for actor observations; critic observations stay undelayed."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg


def add_sensor_latency(cfg: ManagerBasedRlEnvCfg) -> None:
  """Delay IMU by 0–1 control steps and joint velocity by one control step.

  At 50 Hz these are 0–20 ms and 20 ms, respectively. IMU lag is
  resampled every 64 control steps. Keep the delays during playback too.
  """
  for name in ("base_ang_vel", "projected_gravity", "joint_vel"):
    # Actor and critic may share terms in base configurations.
    term = deepcopy(cfg.observations["actor"].terms[name])
    term.delay_min_lag = 1 if name == "joint_vel" else 0
    term.delay_max_lag = 1
    term.delay_update_period = 0 if name == "joint_vel" else 64
    cfg.observations["actor"].terms[name] = term
