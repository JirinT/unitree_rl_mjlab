"""Shared opt-in actuator selection for MiniPawo training and playback."""
import math
from dataclasses import fields, replace


def configure_actuators(task_id, env_cfg, model="builtin", voltage=7.4, friction_model="m6",
                        command_delay_min=3, command_delay_max=6):
    if model == "builtin":
        return
    if model != "bam-sts3215" or task_id != "MiniPawo-Flat":
        raise ValueError("BAM STS3215 is restricted to MiniPawo-Flat")
    if not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("bam_voltage must be finite and positive")
    if friction_model not in {"m1", "m2", "m3", "m4", "m5", "m6"}:
        raise ValueError("Expected BAM friction model m1 through m6")
    if (not isinstance(command_delay_min, int) or not isinstance(command_delay_max, int)
            or not 0 <= command_delay_min <= command_delay_max):
        raise ValueError("BAM command delay must satisfy 0 <= min <= max (physics steps)")
    from src.vendor.bam.mjlab import BamActuatorCfg, bam_init
    from mjlab.managers.event_manager import EventTermCfg
    robot_cfg = env_cfg.scene.entities["robot"]
    actuator = BamActuatorCfg(
        motor_name="feetech_sts3215_7_4V", model=friction_model,
        target_names_expr=(".*",), vin=voltage,
    )
    if command_delay_max:
        if "delay_max_lag" in {field.name for field in fields(actuator)}:
            # Newer mjlab has command delays built into every actuator.
            actuator = replace(actuator, delay_min_lag=command_delay_min,
                               delay_max_lag=command_delay_max)
        else:
            # Older mjlab exposes the same physics-step delay as a wrapper.
            from mjlab.actuator import DelayedActuatorCfg
            actuator = DelayedActuatorCfg(
                base_cfg=actuator, delay_target="position",
                delay_min_lag=command_delay_min, delay_max_lag=command_delay_max,
            )
    robot_cfg.articulation = replace(robot_cfg.articulation, actuators=(actuator,))
    dt_ms = 1000 * env_cfg.sim.mujoco.timestep
    print(f"[BAM] Position command delay: {command_delay_min}..{command_delay_max} physics steps "
          f"({command_delay_min * dt_ms:g}..{command_delay_max * dt_ms:g} ms)")
    env_cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
    print(f"[BAM EXPERIMENT] STS3215 7.4V fitted {friction_model}, supply={voltage}V; "
          "not identified for Waveshare 22414. Rig-specific q_offset and fitted command_delay are not applied; explicit command-delay range is used.")
