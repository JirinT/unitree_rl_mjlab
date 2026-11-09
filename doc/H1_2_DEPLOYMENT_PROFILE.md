# Optional H1_2 Unitree MuJoCo deployment profile

The existing robot, velocity, HandTracking and ArmIntervention tasks remain
unchanged. A new recipe combines the deployment plant modifier with the existing
arm-intervention modifier:

```python
ARM_INTERVENTION_DEPLOYMENT = Recipe(
  name="arm_intervention_deployment",
  modifiers=(match_unitree_mujoco, add_arm_intervention),
)
```

Registered flat task: **Unitree-H1_2-Flat-ArmIntervention-Deployment**.
Experiment: **h1_2_arm_intervention_deployment**.

## Training and playback

From the training repo, using your existing robotics/training environment:

```bash
python -m scripts.train Unitree-H1_2-Flat-ArmIntervention-Deployment \
  --env.scene.num-envs 4096
```

This starts a separate training run. No existing checkpoint is modified. The
recipe preserves the 108 actor inputs / 27 outputs, joint order, action scales,
arm takeover masks, action-history semantics, posture regularization and
performance-gated arm curriculum. The critic layout is preserved too. It is
therefore compatible in tensor dimensions with existing ArmIntervention weights,
although it changes the plant; use the runner's checkpoint-loading mechanism if
you intentionally warm-start. A successful optimizer resume does not establish
that the changed dynamics are already learned.

Visual evaluation uses the same controls added to `play.py`:

```bash
python -m scripts.play Unitree-H1_2-Flat-ArmIntervention-Deployment \
  --onnx-file logs/rsl_rl/h1_2_arm_intervention_deployment/YOUR_RUN/policy.onnx \
  --arm-control demo --num-envs 1 --seed 42
```

On macOS, use `mjpython` for the native viewer. Replace `--arm-control demo` with
`--keyboard-control True` for manual walking/arm commands. Focus the launching
terminal. See `H1_2_ARM_INTERVENTION.md` for keys and the joint-waypoint API.

## Physical profile

The loader uses the frozen Unitree MuJoCo handless H1_2 XML inspected during the
6 November 2026 transfer audit. It reuses the existing H1_2 STL assets. Keep the
same mesh revision on Linux; identical filenames alone do not certify identical
mesh geometry. Source/provenance is recorded in the XML; its BSD-3-Clause notice
is retained in `DEPLOYMENT_MODEL_LICENSE.txt`.

The XML's SDK motor/sensor declarations are replaced with mjlab actuators and
sensor aliases. Foot/palm observation sites retain the original training offsets
without changing physical geometry. The ArmIntervention actor does not consume
Cartesian palm positions. The original baseline XML is not edited.

| Property | New profile |
|---|---|
| Body masses, inertias, joint axes/ranges | Unitree handless model |
| Foot contacts | Original mesh feet, condim 3, priority 0, friction 1.0 |
| Other collisions/exclusions | Original deployment geometry/masks; no capsule overrides |
| Actuated joint damping | 1.0 passive, additional to controller Kd |
| Actuated joint friction loss | 0.2 |
| Hip/waist armature | 0.025 |
| Knee armature | 0.04 |
| Ankle/shoulder pitch-roll armature | 0.005 |
| Other arm-joint armature | 0.002 |
| Ankle-pitch torque limit | 60 Nm |
| Ankle-roll / shoulder pitch-roll limit | 40 Nm |
| Wrist torque limits | 19 Nm |
| Shoulder-yaw / elbow limit | 18 Nm |
| Hip/waist / knee limits | 200 / 300 Nm |
| Motor controller | Explicit PD with the existing Kp/Kd values |
| Physics | Euler, 0.005 s |
| Policy | 0.02 s, decimation 4 |

Explicit PD is recomputed from simulated joint state each physics substep and the
combined torque is clamped. This replaces the implicit built-in position servo.
The existing normalized-action-to-position scales are deliberately retained;
increasing torque limits does not silently enlarge position commands.

The scene uses the supplied deployment XML's nominal MuJoCo solver settings.
This recipe models deterministic synchronous sampled PD. It does **not** emulate
wall-clock slowdown, DDS callback timing, packet loss, sensor age or races in the
Python validator bridge. Measure and address those separately. Updating the bridge
to calculate PD inside its physics loop produces a clearer common timing target.

## Floating-base choice

The default is an exact match to the **supplied Linux configuration**, including
its inherited floating-base damping 1, friction loss 0.2 and armature 0.1 on all
six root DOFs. Your actuated-joint armature edits are already applied by the
loader. This is not the untouched upstream robot, which uses default armature 0.1
for all joints.

The earlier recommendation to use a physically free root requires changing both
sides together. Do not zero it only in training while leaving drag in deployment.
The modifier exposes a code-level option:

```python
from functools import partial

Recipe(
  name="deployment_free_base",
  modifiers=(partial(match_unitree_mujoco, free_base=True), add_arm_intervention),
)
```

Use this only after also setting `damping="0" frictionloss="0" armature="0"` on
`floating_base_joint` in the Linux XML. No extra task is registered for this
alternative; select/register it intentionally. The exported metadata records the
actual root parameters for either choice.

## Nominal parity before further randomization

Only in this recipe, startup `foot_friction`, `base_com`, and `encoder_bias` events
are removed. This avoids old capsule-geom references and preserves nominal
friction, mass distribution and action offsets while establishing parity.
Existing actor observation noise and training pushes remain. Playback disables
observation corruption and pushes as usual.

Add measured/bounded physical and timing uncertainty as later modifiers once the
nominal systems agree. This first profile does not add arbitrary latency or
re-tune rewards. Standing coverage remains 50%, and overridden arms are still
excluded from conflicting posture/action penalties.

## Export and SDK compatibility

For this profile, the runner exports actual explicit-controller Kp/Kd and effort
limits. Torque motors' compiled gains are 1, so reading those as Kp would produce
incorrect deployment metadata. Original tasks retain their export behavior.

Additional ONNX metadata includes deployment profile, explicit-PD model, mesh
contact profile, physics/policy periods, integrator, passive joint dynamics,
armatures and root dynamics. ArmIntervention metadata remains `intervention_v1`.
The SDK adapter accepts the exported actor with the existing matching gain,
default-pose and action-scale YAML:

```bash
python deploy_arm_intervention.py deploy_arm_intervention.yaml \
  /path/to/new/policy.onnx --validate-only
```

Then use `--arm-control keyboard` or `--arm-control demo` with the separate Linux
validator. That offline check validates the actor contract and controller values;
it does not check the running validator's XML or timing. Keep the supplied Linux
joint-armature edits, passive parameters, collision model, torque limits and 5 ms
physics setting consistent with the selected training profile.

## Validation

CPU tests check old-task isolation, tensor/action compatibility, actual mesh
contacts, joint/root dynamics, effort limits, optional root override, explicit PD
state feedback/clipping and controller-gain metadata:

```bash
python -m unittest discover -s tests -p 'test_h1_2*.py'
python -m unittest discover -s tests -p 'test_play*.py'
```

The implementation was also exercised in a real CPU MuJoCo/Warp environment,
through one PPO update and ONNX export, and through SDK offline validation of that
export. These establish executable compatibility, not learned balance quality or
one-shot hardware transfer. Run the actual Linux DDS/validator benchmark after
training.
