# Drop-in Waveshare model

Copy the fitted parameter JSON here as **model.json**, then restart simulation.
MiniPawo automatically selects this file when `MINI_PAWO_BAM_JSON_PATH` is None.
The JSON must have `"actuator": "waveshare_st3215"` and `"model": "m1"` through
`"m6"`. Use the output of `bam.waveshare.fit_mujoco`, not `st3215.local.json`
(which is a recording configuration). Keep all fitted parameters, including
`pwm_limit`, unchanged. No example/fake fit is installed here.

`hardware_settings` supplies voltage and firmware P gain when each has one
consistent value. Mixed settings require explicit `MINI_PAWO_BAM_VIN` and/or
`MINI_PAWO_BAM_KP_FW` in `minipawo_constants.py`. Explicit constants override
metadata. Without metadata or overrides the identification controller defaults
apply (ST3215: 7.4 V and gain 32); for the supplied recording configuration use
7.7 V and gain 64. Choose settings matching your intended hardware operation.

Run, from the framework root in your robotics environment:

```sh
python scripts/play.py MiniPawo-Flat --agent zero --device cpu --num-envs 1 --viewer viser
```

This is a controller/simulation check with zero policy actions. To play a learned
policy, replace `--agent zero` with your usual checkpoint arguments. Both
training and playback default to `--actuator-model config`; use `bam-waveshare-st3215` for the explicit Waveshare CLI option.

If model.json is absent, the existing bundled STS3215 experimental fit remains
active and startup prints that fallback explicitly. An invalid file raises an
error; it is never silently replaced. The current 3-6 physics-step delay is
preserved. Fitted testbench command_delay/q_offset are not automatically applied.

## Select Waveshare with CLI flags

Save each fitted variant as `m1.json`, `m2.json`, etc. in
`src/vendor/bam/params/waveshare_st3215/`. A matching `model.json` is accepted
when the requested variant file is absent. The JSON's `model` must match the
flag: an M6 fit is not converted into M1 by changing its name.

```sh
mjpython scripts/play.py MiniPawo-Flat \
  --onnx-file logs/rsl_rl/MiniPawo_velocity/10Hz/policy.onnx \
  --actuator-model bam-waveshare-st3215 \
  --bam-friction-model m1 --bam-voltage 7.4 \
  --policy-decimation 20 --action-scale 0.25 \
  --device cpu --num-envs 1 --headless-steps 500

python scripts/train.py MiniPawo-Flat \
  --actuator-model bam-waveshare-st3215 \
  --bam-friction-model m1 --bam-voltage 7.4 \
  --env.decimation 20 \
  --agent.experiment-name MiniPawo_Waveshare_m1_10Hz \
  --agent.logger tensorboard --agent.upload-model False
```

The CLI voltage overrides JSON metadata. Firmware gain comes from consistent
`hardware_settings` metadata (controller default if absent). The existing
`--bam-command-delay-min` and `--bam-command-delay-max` flags also apply, with
defaults of 3 and 6 physics steps. Explicit Waveshare selection errors if a
matching fit is absent; it never falls back to the bundled Feetech motor.
The old `bam-sts3215` selector still refers to that bundled Feetech motor.

### Firmware P gain from the CLI

For explicit `bam-waveshare-st3215` or `bam-sts3215` selection, use
`--bam-kp-fw 64` in both training and playback to override firmware
P gain. This is the servo register gain, not MuJoCo PD stiffness.
If omitted, JSON hardware metadata applies when supported, otherwise
the controller default applies (32 for the STS3215 controller).
In `config` mode use `MINI_PAWO_BAM_KP_FW` in the constants file.
