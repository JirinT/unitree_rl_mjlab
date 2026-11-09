"""Deployment-plant checks; no DDS, viewer or GPU is required."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

import src.tasks
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.assets.robots.unitree_h1_2.deployment import (
  DEPLOYMENT_PROFILE, deployment_metadata, get_deployment_robot_cfg, get_deployment_spec,
)
from src.assets.robots.unitree_h1_2.h1_2_constants import get_h1_2_robot_cfg
from src.tasks.velocity.config.h1_2.modifiers.match_unitree_mujoco import match_unitree_mujoco
from src.tasks.velocity.mdp.arm_intervention import ArmInterventionActionCfg

TASK = "Unitree-H1_2-Flat-ArmIntervention-Deployment"


class Tests(unittest.TestCase):
  def test_recipe_retains_contract_and_isolates_original_tasks(self):
    original = load_env_cfg("Unitree-H1_2-Flat-ArmIntervention")
    original_robot = asdict(original.scene.entities["robot"])
    new = load_env_cfg(TASK)
    self.assertIsInstance(new.actions["joint_pos"], ArmInterventionActionCfg)
    self.assertEqual(asdict(new.actions["joint_pos"]), asdict(original.actions["joint_pos"]))
    self.assertEqual({k: asdict(v) for k, v in new.observations.items()}, {k: asdict(v) for k, v in original.observations.items()})
    self.assertEqual({k: asdict(v) for k, v in new.rewards.items()}, {k: asdict(v) for k, v in original.rewards.items()})
    self.assertEqual({k: asdict(v) for k, v in new.curriculum.items()}, {k: asdict(v) for k, v in original.curriculum.items()})
    self.assertEqual(original_robot, asdict(load_env_cfg("Unitree-H1_2-Flat-ArmIntervention").scene.entities["robot"]))
    self.assertEqual(load_rl_cfg(TASK).experiment_name, "h1_2_arm_intervention_deployment")
    for name in ("Unitree-H1_2-Flat", "Unitree-H1_2-Flat-HandTracking", "Unitree-H1_2-Rough-ArmIntervention"):
      cfg = load_env_cfg(name)
      self.assertFalse(any(isinstance(a, IdealPdActuatorCfg) for a in cfg.scene.entities["robot"].articulation.actuators))
      self.assertIn("foot_friction", cfg.events)
      self.assertEqual(cfg.sim.mujoco.integrator, "implicitfast")

  def test_profile_timing_nominal_events_and_playback(self):
    for play in (False, True):
      cfg = load_env_cfg(TASK, play=play)
      self.assertEqual(cfg.sim.mujoco.integrator, "euler")
      self.assertEqual(cfg.sim.mujoco.timestep * cfg.decimation, 0.02)
      self.assertEqual(cfg.sim.mujoco.iterations, 100)
      self.assertEqual(cfg.sim.mujoco.ls_iterations, 50)
      for name in ("foot_friction", "encoder_bias", "base_com"):
        self.assertNotIn(name, cfg.events)
      self.assertEqual(cfg.observations["actor"].enable_corruption, not play)
      self.assertEqual("arm_intervention" in cfg.curriculum, not play)
      if play:
        self.assertEqual(cfg.curriculum, {})
        self.assertNotIn("push_robot", cfg.events)

  def test_passive_dynamics_armatures_and_torque_limits_match_linux_model(self):
    robot = Entity(get_deployment_robot_cfg())
    model = robot.spec.compile()
    self.assertEqual((model.njnt, model.nu), (28, 27))
    self.assertAlmostEqual(model.body_mass.sum(), 66.984)
    np.testing.assert_allclose(model.dof_damping, 1.)
    np.testing.assert_allclose(model.dof_frictionloss, 0.2)
    np.testing.assert_allclose(model.dof_armature[:6], 0.1)
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    self.assertEqual(names, list(robot.joint_names))
    expected = {"left_hip_pitch_joint": (0.025, 200.), "left_knee_joint": (0.04, 300.),
                "left_ankle_pitch_joint": (0.005, 60.), "left_ankle_roll_joint": (0.005, 40.),
                "left_shoulder_pitch_joint": (0.005, 40.), "left_elbow_joint": (0.002, 18.),
                "left_wrist_pitch_joint": (0.002, 19.)}
    for name, (armature, limit) in expected.items():
      joint = model.joint(name).id
      self.assertAlmostEqual(model.dof_armature[model.jnt_dofadr[joint]], armature)
      actuator = names.index(name)
      np.testing.assert_allclose(model.actuator_forcerange[actuator], [-limit, limit])
    # All are opaque torque motors; stiffness/damping are in explicit PD code.
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1.)
    np.testing.assert_allclose(model.actuator_biasprm, 0.)

  def test_mesh_contacts_and_observation_sites_are_preserved(self):
    model = Entity(get_deployment_robot_cfg()).spec.compile()
    for side in ("left", "right"):
      body = model.body(f"{side}_ankle_roll_link").id
      contacts = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and model.geom_contype[i]]
      self.assertEqual(len(contacts), 1)
      i = contacts[0]
      self.assertEqual(model.geom_type[i], mujoco.mjtGeom.mjGEOM_MESH)
      self.assertEqual(model.geom_condim[i], 3)
      self.assertEqual(model.geom_priority[i], 0)
      self.assertAlmostEqual(model.geom_friction[i, 0], 1.)
      self.assertGreaterEqual(model.site(f"{side}_foot").id, 0)
      self.assertGreaterEqual(model.site(f"{side}_palm").id, 0)
    for name in ("imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"):
      self.assertGreaterEqual(model.sensor(name).id, 0)
    self.assertEqual(model.nexclude, 0)  # No old capsule-contact exclusions.

  def test_joint_contract_still_matches_existing_sdk_geometry(self):
    baseline = Entity(get_h1_2_robot_cfg()).spec.compile()
    deployment = Entity(get_deployment_robot_cfg()).spec.compile()
    for i in range(1, baseline.njnt):
      name = baseline.joint(i).name
      j = deployment.joint(name).id
      np.testing.assert_allclose(deployment.jnt_range[j], baseline.jnt_range[i])
      np.testing.assert_allclose(deployment.jnt_axis[j], baseline.jnt_axis[i])
    for name in ("left_palm", "right_palm", "left_foot", "right_foot", "imu"):
      np.testing.assert_allclose(deployment.site(name).pos, baseline.site(name).pos)

  def test_free_base_option_does_not_mutate_nominal_or_hinges(self):
    free = Entity(get_deployment_robot_cfg(free_base=True)).spec.compile()
    nominal = Entity(get_deployment_robot_cfg()).spec.compile()
    for field in ("dof_damping", "dof_frictionloss", "dof_armature"):
      np.testing.assert_allclose(getattr(free, field)[:6], 0.)
      np.testing.assert_allclose(getattr(free, field)[6:], getattr(nominal, field)[6:])
    cfg = load_env_cfg("Unitree-H1_2-Flat-ArmIntervention")
    match_unitree_mujoco(cfg, free_base=True)
    model = Entity(cfg.scene.entities["robot"]).spec.compile()
    np.testing.assert_allclose(model.dof_armature[:6], 0.)

  def test_export_metadata_contains_controller_gains_instead_of_motor_gain_one(self):
    robot = Entity(get_deployment_robot_cfg())
    model = robot.spec.compile()
    # The actual environment prefixes joint names by scene entity name.
    wrapped = SimpleNamespace(
      jnt_dofadr=model.jnt_dofadr, dof_damping=model.dof_damping,
      dof_armature=model.dof_armature, dof_frictionloss=model.dof_frictionloss,
      joint=lambda name: model.joint(name.removeprefix("robot/")),
    )
    env = SimpleNamespace(scene={"robot": robot}, sim=SimpleNamespace(mj_model=wrapped),
                          cfg=load_env_cfg(TASK), step_dt=.02)
    meta = deployment_metadata(env)
    self.assertEqual(meta["deployment_profile"], DEPLOYMENT_PROFILE)
    self.assertEqual(meta["actuator_model"], "explicit_pd")
    self.assertEqual(len(meta["joint_stiffness"]), 27)
    for name, kp, kd, limit in (("left_knee_joint", 157.7, 10.1, 300.),
                                ("left_ankle_pitch_joint", 19.7, 1.3, 60.),
                                ("left_wrist_pitch_joint", 7.9, .5, 19.)):
      i = list(robot.joint_names).index(name)
      self.assertEqual(meta["joint_stiffness"][i], kp)
      self.assertEqual(meta["joint_damping"][i], kd)
      self.assertEqual(meta["joint_effort_limits"][i], limit)
    np.testing.assert_allclose(meta["floating_base_damping"], 1.)
    env.scene["robot"] = Entity(get_h1_2_robot_cfg())
    self.assertEqual(deployment_metadata(env), {})

  def test_explicit_pd_uses_fresh_state_and_clamps_combined_torque(self):
    cmd = SimpleNamespace(position_target=torch.tensor([[2., 0.]]), pos=torch.tensor([[0., 1.]]),
                          velocity_target=torch.zeros(1, 2), vel=torch.tensor([[3., -2.]]),
                          effort_target=torch.zeros(1, 2))
    params = {"stiffness": torch.tensor([[10., 10.]]), "damping": torch.tensor([[1., 1.]]),
              "force_limit": torch.tensor([[5., 5.]])}
    actuator = Entity(get_deployment_robot_cfg()).actuators[0]
    for name, value in params.items():
      setattr(actuator, name, value)
    result = actuator.compute(cmd)
    torch.testing.assert_close(result, torch.tensor([[5., -5.]]))
    cmd.pos = torch.tensor([[2., 0.]])
    torch.testing.assert_close(actuator.compute(cmd), torch.tensor([[-3., 2.]]))


if __name__ == "__main__":
  unittest.main()
