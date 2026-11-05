import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    joint_action = self.env.unwrapped.action_manager.get_term("joint_pos")
    if hasattr(joint_action, "arm_action_ids"):
      # Exported policy contains no random generator or arm override controller.
      metadata.update({
        "arm_control_contract": "intervention_v1",
        "arm_joint_names": [joint_action.target_names[i] for i in joint_action.arm_action_ids.tolist()],
        "arm_action_indices": joint_action.arm_action_ids.tolist(),
        "arm_control_observation": "mask_left,mask_right,executed_arm_target_minus_default[14]",
        "previous_action_semantics": "executed_target_minus_offset_div_scale",
        "arm_max_joint_speed": joint_action.cfg.max_joint_speed,
        "arm_max_joint_acceleration": joint_action.cfg.max_joint_acceleration,
        "arm_handover_time": joint_action.cfg.handover_time,
      })
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
