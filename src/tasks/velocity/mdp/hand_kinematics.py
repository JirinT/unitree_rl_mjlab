"""Batched Torch forward kinematics using static MJCF transforms, without IK."""

import mujoco
import torch

from mjlab.utils.lab_api.math import quat_apply, quat_mul


class PalmKinematics:
  """Evaluate the pelvis-to-palm chains from measured absolute joint angles.

  MuJoCo compiles the geometry once. Runtime FK stays on the observation device;
  no per-environment CPU kinematics or physics integration is performed.
  """

  def __init__(self, model, joint_names, site_names, device, root_name="pelvis"):
    self.num_joints = len(joint_names)
    self.root = model.body(root_name).id
    self.sites = [model.site(name).id for name in site_names]
    ancestors = set()
    for site in self.sites:
      body = int(model.site_bodyid[site])
      while body != self.root:
        if body == 0:
          raise ValueError("Palm sites must descend from the configured pelvis.")
        ancestors.add(body)
        body = int(model.body_parentid[body])
    joint_map = {name: i for i, name in enumerate(joint_names)}
    def tensor(value):
      return torch.as_tensor(value.copy(), dtype=torch.float32, device=device)
    self.nodes = []
    for body in sorted(ancestors):
      joints = []
      for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
          raise ValueError("Palm FK currently supports hinge joints along the arm chains.")
        name = model.joint(j).name
        if name not in joint_map:
          raise ValueError(f"Missing measured joint in palm chain: {name}")
        joints.append((joint_map[name], tensor(model.jnt_axis[j]), tensor(model.jnt_pos[j]),
                       float(model.qpos0[model.jnt_qposadr[j]])))
      self.nodes.append((body, int(model.body_parentid[body]), tensor(model.body_pos[body]),
                         tensor(model.body_quat[body]), joints))
    self.site_transforms = [(int(model.site_bodyid[s]), tensor(model.site_pos[s])) for s in self.sites]

  def __call__(self, joint_pos):
    if joint_pos.ndim != 2 or joint_pos.shape[1] != self.num_joints:
      raise ValueError("Palm FK expects all robot joints in entity joint order.")
    count = joint_pos.shape[0]
    zero = joint_pos.new_zeros(count, 3)
    identity = joint_pos.new_zeros(count, 4)
    identity[:, 0] = 1
    transforms = {self.root: (zero, identity)}
    for body, parent, offset, orientation, joints in self.nodes:
      parent_pos, parent_quat = transforms[parent]
      pos = parent_pos + quat_apply(parent_quat, offset.expand(count, -1))
      quat = quat_mul(parent_quat, orientation.expand(count, -1))
      for index, axis, anchor, reference in joints:
        half = (joint_pos[:, index] - reference) * 0.5
        rotation = torch.cat((half.cos()[:, None], half.sin()[:, None] * axis), dim=1)
        anchor = anchor.expand(count, -1)
        pos = pos + quat_apply(quat, anchor - quat_apply(rotation, anchor))
        quat = quat_mul(quat, rotation)
      transforms[body] = (pos, quat)
    return torch.stack([
      transforms[body][0] + quat_apply(transforms[body][1], local.expand(count, -1))
      for body, local in self.site_transforms
    ], dim=1)
