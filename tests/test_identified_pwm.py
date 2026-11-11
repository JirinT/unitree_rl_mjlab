"""Offline checks for identified PWM physics and recipe controller gains."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.vendor.bam.actuator import TorchBackend
from src.vendor.bam.model import load_model
from src.vendor.bam.mjlab import BamActuator, BamActuatorCfg
import src.tasks
from mjlab.tasks.registry import load_env_cfg

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "src/vendor/bam/params/esp_pwm_st3215/m1.json"

class Tests(unittest.TestCase):
    def test_controller_matches_esp_including_clipping_numpy_and_torch(self):
        q = np.array([[.1, -.1, .3], [.2, -.2, .1]])
        target = np.array([[.12, -.12, 20], [.2, -20, .12]])
        vel = np.array([[.04, -.04, 0], [.1, .1, -.1]])
        expected = 10.9 * np.clip((4*(target-q)-.6*vel)/(2*np.pi/4096*1000), -1, 1)
        for use_torch in (False, True):
            model = load_model(str(FIT))
            model.controller_kd.value = .6
            act = model.actuator
            act.vin, act.kp = 10.9, 4
            values = (target, q, vel)
            if use_torch:
                act.backend = TorchBackend()
                values = tuple(torch.tensor(x) for x in values)
            control = act.compute_control(*values, .005)
            np.testing.assert_allclose(np.asarray(control), expected)
            torque = act.compute_torque(control, True, values[1], values[2])
            np.testing.assert_allclose(np.asarray(torque), model.kt.value*(expected-model.kt.value*vel)/model.R.value)
        self.assertEqual(json.loads(FIT.read_text())["controller_kd"], .2)

    def test_recipe_settings_and_sensor_composition(self):
        for suffix in ("", "-ActionScale", "-SensorLatency-ActionScale",
                       "-SensorLatency-ActuatorLatency",
                       "-SensorLatency-ActuatorLatency-ActionScale"):
            cfg = load_env_cfg("MiniPawo-Flat-IdentifiedPWM"+suffix, play=True)
            actuator, = cfg.scene.entities["robot"].articulation.actuators
            if "ActuatorLatency" in suffix:
                self.assertGreater(actuator.delay_max_lag, 0)
            actuator = getattr(actuator, "base_cfg", actuator)
            self.assertIsInstance(actuator, BamActuatorCfg)
            self.assertEqual((actuator.kp_fw, actuator.kd_fw, actuator.vin), (4, .6, 10.9))
            self.assertEqual(Path(actuator.json_path), FIT)
            self.assertIn("bam_init", cfg.events)
            if "ActionScale" in suffix:
                self.assertEqual(cfg.actions["joint_pos"].scale, .25)
            if "SensorLatency" in suffix:
                self.assertGreater(cfg.observations["actor"].terms["joint_vel"].delay_max_lag, 0)

    def test_invalid_derivative_gains(self):
        for value in (-1, float("nan"), 2.56):
            with self.assertRaises(ValueError):
                BamActuatorCfg(json_path=str(FIT), target_names_expr=(".*",), kd_fw=value)

    def test_runtime_loads_override_and_fitted_physics(self):
        from mjlab.envs import ManagerBasedRlEnv
        cfg = load_env_cfg("MiniPawo-Flat-IdentifiedPWM-ActionScale", play=True)
        cfg.scene.num_envs = 2
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        try:
            act, = env.scene["robot"]._actuators
            self.assertIsInstance(act, BamActuator)
            self.assertEqual(act._bam_model.controller_kd.value, .6)
            fit = json.loads(FIT.read_text())
            for key in ("kt", "R", "armature", "friction_base", "friction_viscous"):
                self.assertEqual(getattr(act._bam_model, key).value, fit[key])
            env.reset()
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(torch.zeros((2,10)))
                self.assertTrue(torch.isfinite(reward).all())
                self.assertTrue(torch.isfinite(env.scene["robot"].data.joint_pos).all())
        finally:
            env.close()

if __name__ == "__main__":
    unittest.main()
