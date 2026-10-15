"""Shared opt-in actuator selection for MiniPawo training and playback."""
import math
from dataclasses import replace


def configure_actuators(task_id, env_cfg, model="builtin", voltage=7.4, friction_model="m6"):
    if model == "builtin":
        return
    if model != "bam-sts3215" or task_id != "MiniPawo-Flat":
        raise ValueError("BAM STS3215 is restricted to MiniPawo-Flat")
    if not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("bam_voltage must be finite and positive")
    if friction_model not in {"m1", "m2", "m3", "m4", "m5", "m6"}:
        raise ValueError("Expected BAM friction model m1 through m6")
    from src.vendor.bam.mjlab import BamActuatorCfg, bam_init
    from mjlab.managers.event_manager import EventTermCfg
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.articulation = replace(
        robot_cfg.articulation,
        actuators=(BamActuatorCfg(
            motor_name="feetech_sts3215_7_4V", model=friction_model,
            target_names_expr=(".*",), vin=voltage,
        ),),
    )
    env_cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
    print(f"[BAM EXPERIMENT] STS3215 7.4V fitted {friction_model}, supply={voltage}V; "
          "not identified for Waveshare 22414. Rig-specific q_offset and command_delay are not applied.")
