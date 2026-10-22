# Copyright 2026 Theo Moore-Calters

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

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
