BAM source subset from https://github.com/Rhoban/bam, commit e9a619d56da5236206f4de6ceec2c1ee1b497b5c (Apache-2.0; LICENSE included).
Includes original core, STS3215 controller and fitted m1–m6 JSONs. Local modifications: imports namespaced under src.vendor.bam; registry narrowed to STS3215; mjlab.py omits standalone pendulum simulator and unused imports; CommandField annotation changed to str for mjlab 1.2. Other compatibility changes, if needed, are documented in the integration README.
No global package installation or environment upgrade. Requires existing numpy, torch and mjlab dependencies.

Fixed upstream mjlab M6 quadratic friction masking to match Model.compute_frictions: apply only for opposing torque signs and exclude equal-magnitude drive/backdrive cases. Covered by NumPy/Torch parity tests for m1–m6.

Removed colorama dependency from message.py; formatting helpers now emit plain text.

Joint damping initialization accepts scalar MuJoCo APIs and falls back to the 3.7 vector API, in both existing-actuator and new-motor paths.

Waveshare controller integration: copied waveshare/actuator.py from the local
identification project /Users/jiri/Desktop/school/Diplomka/UnitreeH1_2_uni_lab/bam_servo_identification/bam/waveshare/actuator.py, SHA-256
71a3c2885c83f5f8ccf035c187cc41ed5509c00c3d573e7da97e21424150a64f.
Copyright 2026 Theo Moore-Calters, Apache-2.0. Only package imports changed;
ST3215 PWM-limit parameter and ST3025 controller equations are retained.
Registry includes waveshare_st3215 and waveshare_st3025. Custom Waveshare
configs read consistent hardware_settings voltage/gain metadata; explicit
config overrides take precedence. Conflicting metadata requires an override.
