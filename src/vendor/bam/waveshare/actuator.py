# Copyright 2026 Theo Moore-Calters

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import math
import numpy as np

from src.vendor.bam.actuator import VoltageControlledActuator
from src.vendor.bam.parameter import Parameter
from src.vendor.bam.testbench import Testbench
from src.vendor.bam.feetech.actuator import STS3215Actuator


class ST3025Actuator(VoltageControlledActuator):
    """Waveshare ST3025 servo operated at 12 V."""

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=12.0,
            kp=32,
            # Inherited from STS3215; error_gain_ratio is fitted for ST3025.
            error_gain=0.166,
            max_pwm=1.0,  # BAM default; ST3025 limit not measured
        )

    def initialize(self):
        # Datasheet: 9 kgf.cm/A = 0.8825985 Nm/A.
        self.model.kt = Parameter(0.8825985, 0.05, 2.5)
        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)
        # Estimated from 12 V / 4.4 A locked-rotor current.
        self.model.R = Parameter(2.7272727, 0.1, 10.0)
        self.model.armature = Parameter(0.0001, 0.00001, 0.04)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def compute_control(self, q_target, q, dq, dt):
        duty_cycle = (
            (q_target - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle

        return self.vin * duty_cycle

class ST3215Actuator(STS3215Actuator):
    """ST3215 candidate model, using STS3215 firmware behavior as a hypothesis.

    The voltage, gain and PWM conversion are NOT measurements of this unit.
    Verify P-only control and the target rate limiter against the real firmware.
    Motor/friction parameters and target speed are fitted from the recordings.
    """

    def initialize(self):
        super().initialize()
        # Save the PWM cap in parameter JSON, so mjlab uses the same cap.
        self.model.pwm_limit = Parameter(0.97, 0.01, 1.0, optimize=False)

    def load_log(self, log):
        super().load_log(log)
        if "max_pwm" in log:
            self.model.pwm_limit.value = log["max_pwm"]

    def compute_control(self, q_target, q, dq, dt):
        self.max_pwm = self.model.pwm_limit.value
        return super().compute_control(q_target, q, dq, dt)


class ST3215PwmActuator(VoltageControlledActuator):
    """ST3215 driven by the external ESP PD-to-PWM controller.

    Uses calibrated joint radians, not raw motor polarity. The controller has
    no servo-position rate limiter. PWM +/-1000 is assumed to represent full
    bridge duty; verify this conversion and the actual supply voltage on hardware.
    Motor constants below are identification seeds, not an identified model.
    Sampling, target clamps, bus delay and PWM integer quantization are not
    reproduced by this continuous control law.
    """

    stateful = False
    RAD_PER_TICK = 2.0 * math.pi / 4096
    PWM_FULL_SCALE = 1000.0

    def __init__(self, testbench_class: Testbench, vin=12.0, kp=4.0, kd=0.2):
        self._initial_kd = kd
        super().__init__(
            testbench_class, vin=vin, kp=kp,
            error_gain=1.0 / (self.RAD_PER_TICK * self.PWM_FULL_SCALE),
            max_pwm=1.0,
        )

    def initialize(self):
        # Output-referred initial guesses and search bounds, not fitted values.
        self.model.kt = Parameter(1.0, 0.05, 2.5)
        self.model.R = Parameter(5.0, 0.1, 20.0)
        self.model.armature = Parameter(0.015, 0.00001, 0.08)
        # Hardware settings must not be optimized as motor/friction parameters.
        self.model.controller_kd = Parameter(self._initial_kd, 0.0, 2.55, optimize=False)
        self.model.pwm_limit = Parameter(1.0, 0.0, 1.0, optimize=False)

    @staticmethod
    def _check_setting(name, value, low, high=None, positive=False):
        values = np.asarray(value, dtype=float)
        if (not np.isfinite(values).all() or (values < low).any()
                or (high is not None and (values > high).any())
                or (positive and (values <= 0).any())):
            raise ValueError(f"Invalid ESP PWM log setting {name}: {value}")

    def load_log(self, log):
        # Avoid fitting the external controller to old internal-servo recordings.
        motor = np.asarray(log.get("motor", "esp_pwm_st3215"))
        if not np.all(motor == "esp_pwm_st3215"):
            raise ValueError("ESP PWM identification requires esp_pwm_st3215 recordings")
        for key in ("kp", "kd", "vin", "max_pwm"):
            if key not in log:
                raise ValueError(f"ESP PWM recordings must include {key} metadata")
        self._check_setting("kp", log["kp"], 0, 25.5)
        self._check_setting("kd", log["kd"], 0, 2.55)
        self._check_setting("vin", log["vin"], 0, positive=True)
        self._check_setting("max_pwm", log["max_pwm"], 0, 1, positive=True)
        super().load_log(log)
        self.model.controller_kd.value = log["kd"]
        self.model.pwm_limit.value = log["max_pwm"]

    def get_extra_inertia(self):
        return self.model.armature.value

    def compute_control(self, q_target, q, dq, dt):
        # Same joint-coordinate PD equation as the ESP, normalized from PWM units.
        duty = self.error_gain * (
            self.kp * (q_target - q) - self.model.controller_kd.value * dq
        )
        self.duty_cycle = self.backend.clamp(
            duty, -self.model.pwm_limit.value, self.model.pwm_limit.value
        )
        return self.vin * self.duty_cycle
