BAM source subset from https://github.com/Rhoban/bam, commit e9a619d56da5236206f4de6ceec2c1ee1b497b5c (Apache-2.0; LICENSE included).
Includes original core, STS3215 controller and fitted m1–m6 JSONs. Local modifications: imports namespaced under src.vendor.bam; registry narrowed to STS3215; mjlab.py omits standalone pendulum simulator and unused imports; CommandField annotation changed to str for mjlab 1.2. Other compatibility changes, if needed, are documented in the integration README.
No global package installation or environment upgrade. Requires existing numpy, torch, colorama and mjlab dependencies.

Fixed upstream mjlab M6 quadratic friction masking to match Model.compute_frictions: apply only for opposing torque signs and exclude equal-magnitude drive/backdrive cases. Covered by NumPy/Torch parity tests for m1–m6.
