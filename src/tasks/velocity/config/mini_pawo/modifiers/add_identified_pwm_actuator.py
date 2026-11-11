"""Identified ST3215 M1 physics with the external ESP PWM-PD controller."""
from dataclasses import replace
from mjlab.managers.event_manager import EventTermCfg
from src import SRC_PATH
from src.assets.robots.mini_pawo.minipawo_constants import make_mini_pawo_bam_actuator
from src.vendor.bam.mjlab import bam_init

# Controller settings are separate from the motor/friction identification.
PWM_KP = 4.0
PWM_KD = 0.6
PWM_VOLTAGE = 7.5  # Match the real supply; fitting recordings were 10.9-11 V.
PWM_MODEL_PATH = SRC_PATH / "vendor/bam/params/esp_pwm_st3215/m1.json"


def add_identified_pwm_actuator(cfg) -> None:
    robot = cfg.scene.entities["robot"]
    actuator = make_mini_pawo_bam_actuator(
        json_path=str(PWM_MODEL_PATH), vin=PWM_VOLTAGE, kp_fw=PWM_KP, kd_fw=PWM_KD,
    )
    robot.articulation = replace(robot.articulation, actuators=(actuator,))
    cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
