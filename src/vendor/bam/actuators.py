# STS3215-only registry for the project-local BAM subset.
from .testbench import Pendulum
from .feetech.actuator import STS3215Actuator
actuators = {"sts3215": lambda: STS3215Actuator(Pendulum)}
