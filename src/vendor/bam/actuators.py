# Controller registry for the project-local BAM subset.
from .testbench import Pendulum
from .feetech.actuator import STS3215Actuator
from .waveshare.actuator import ST3025Actuator, ST3215Actuator

actuators = {
    "sts3215": lambda: STS3215Actuator(Pendulum),
    "waveshare_st3215": lambda: ST3215Actuator(Pendulum),
    "waveshare_st3025": lambda: ST3025Actuator(Pendulum),
}
