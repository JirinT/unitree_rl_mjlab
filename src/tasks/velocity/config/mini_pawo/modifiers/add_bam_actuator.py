"""Bundled Feetech STS3215 M6 servo model, without added command latency."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg

from src.assets.robots.mini_pawo.minipawo_constants import (
  make_mini_pawo_bam_actuator,
)
from src.vendor.bam.mjlab import bam_init


def add_bam_actuator(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use BAM's bundled 7.4 V Feetech fit and default firmware gain.

  Select the bundled motor explicitly, independently of the Waveshare JSON
  configured in robot constants. Command delays are a separate experiment.
  """
  robot = cfg.scene.entities["robot"]
  actuator = make_mini_pawo_bam_actuator(
    motor_name="feetech_sts3215_7_4V",
    model="m6",
    vin=7.4,
    kp_fw=32,
    delay_min_lag=0,
    delay_max_lag=0,
  )
  robot.articulation = replace(robot.articulation, actuators=(actuator,))
  cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
