"""Legacy CLI overrides and BAM startup registration for MiniPawo."""
import json
import math
from pathlib import Path

from src import SRC_PATH
from dataclasses import replace


def configure_actuators(task_id, env_cfg, model="config", voltage=12, friction_model="m6",
                        command_delay_min=3, command_delay_max=6, *, kp_fw=None):
    # "config" preserves the actuator selected in minipawo_constants.py.
    if model not in {"config", "builtin", "bam-sts3215", "bam-waveshare-st3215"}:
        raise ValueError(f"Unknown actuator selection: {model}")
    if task_id != "MiniPawo-Flat":
        if model in {"bam-sts3215", "bam-waveshare-st3215"}:
            raise ValueError("BAM CLI override is restricted to MiniPawo-Flat")
        return {}
    from src.assets.robots.mini_pawo.minipawo_constants import (
        MINI_PAWO_BUILTIN_ACTUATOR_LEG, make_mini_pawo_bam_actuator,
    )
    from src.vendor.bam.mjlab import bam_init
    from mjlab.managers.event_manager import EventTermCfg

    robot_cfg = env_cfg.scene.entities["robot"]
    if model == "builtin":
        robot_cfg.articulation = replace(
            robot_cfg.articulation, actuators=(MINI_PAWO_BUILTIN_ACTUATOR_LEG,))
    elif model in {"bam-sts3215", "bam-waveshare-st3215"}:
        if not math.isfinite(voltage) or voltage <= 0:
            raise ValueError("bam_voltage must be finite and positive")
        if kp_fw is not None and (not math.isfinite(kp_fw) or kp_fw < 0):
            raise ValueError("bam_kp_fw must be finite and nonnegative")
        if friction_model not in {"m1", "m2", "m3", "m4", "m5", "m6"}:
            raise ValueError("Expected BAM friction model m1 through m6")
        source = (
            {"json_path": waveshare_parameter_path(friction_model)}
            if model == "bam-waveshare-st3215" else
            {"motor_name": "feetech_sts3215_7_4V", "model": friction_model}
        )
        actuator = make_mini_pawo_bam_actuator(
            **source, vin=voltage, kp_fw=kp_fw,
            delay_min_lag=command_delay_min, delay_max_lag=command_delay_max,
        )
        robot_cfg.articulation = replace(robot_cfg.articulation, actuators=(actuator,))
    info = describe_actuators(robot_cfg)
    if info["actuator_model"] == "bam":
        env_cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
    else:
        env_cfg.events.pop("bam_init", None)
    print(f"[MiniPawo actuators] {info}")
    return info


def describe_actuators(robot_cfg):
    """Report the resolved config, including a possible mjlab 1.2 delay wrapper."""
    from src.vendor.bam.mjlab import BamActuatorCfg

    actuator, = robot_cfg.articulation.actuators
    base = getattr(actuator, "base_cfg", actuator)
    if not isinstance(base, BamActuatorCfg):
        return {"actuator_model": "builtin"}
    with open(base._resolved_json_path) as stream:
        parameters = json.load(stream)
    return dict(
        actuator_model="bam", bam_json_path=base._resolved_json_path,
        bam_actuator=parameters["actuator"],
        bam_friction_model=parameters["model"], bam_voltage=base.vin, bam_kp_fw=base.kp_fw,
        bam_command_delay_min=getattr(actuator, "delay_min_lag", 0),
        bam_command_delay_max=getattr(actuator, "delay_max_lag", 0),
    )


def waveshare_parameter_path(friction_model):
    """Select the requested fitted variant; never reinterpret another fit."""
    directory = SRC_PATH / "vendor" / "bam" / "params" / "waveshare_st3215"
    variant = directory / f"{friction_model}.json"
    path = variant if variant.is_file() else directory / "model.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No Waveshare {friction_model} fit. Copy your fitted JSON to {variant} "
            f"(or {path} with model={friction_model!r})."
        )
    with path.open() as stream:
        parameters = json.load(stream)
    if parameters.get("actuator") not in {"sts3215", "waveshare_st3215"}:
        raise ValueError(f"{path} must contain actuator='sts3215' or 'waveshare_st3215'")
    if parameters.get("model") != friction_model:
        raise ValueError(
            f"Requested {friction_model}, but {path} contains model={parameters.get('model')!r}. "
            "Use the matching --bam-friction-model or supply the fitted variant."
        )
    return path
