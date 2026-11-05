"""Named combinations of changes to the H1_2 velocity baseline."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from .modifiers.add_hand_tracking import add_hand_tracking
from .modifiers.add_arm_intervention import add_arm_intervention

# A modifier edits its own environment configuration in place.
Modifier = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class Recipe:
  name: str
  modifiers: tuple[Modifier, ...] = ()


BASELINE = Recipe(name="baseline")


HAND_TRACKING = Recipe(
  name="hand_tracking",
  modifiers=(add_hand_tracking,),
)


ARM_INTERVENTION = Recipe(
  name="arm_intervention",
  modifiers=(add_arm_intervention,),
)
