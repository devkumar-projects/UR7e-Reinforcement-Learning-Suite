# Frozen SAC trajectory following: source and evaluation

This directory makes the existing UR7e-labelled simple-environment checkpoint reproducible on CPU and audits the original completion metric. It contains the portable environment, fixed DLS comparison, evaluation protocol, strict endpoint checks and actor export.

## Published artifacts

- [Hugging Face model, original checkpoint and portable actor](https://huggingface.co/devkumarhf/ur7e-sac-line-following)
- [Live browser policy lab](https://huggingface.co/spaces/devkumarhf/UR7e-Policy-Lab)
- [Kaggle experimental analysis](https://www.kaggle.com/code/devkumarkaggle/ur7e-sac-vs-dls-reproducible-policy-benchmark)
- [Measured episode dataset](https://www.kaggle.com/datasets/devkumarkaggle/ur7e-sac-vs-dls-seeded-kinematic-benchmark)

GitHub maintains implementation and reproduction instructions. Hugging Face distributes versioned weights and inference artifacts. Kaggle provides the measured dataset and executable analysis.

## Reproduce

From this directory, install requirements in an isolated Python environment; use a CPU PyTorch build if needed. The loader finds the original checkpoint at `../../line-follower-simulation-simple/models_traj/best_model.zip` when no local copy is present.

```sh
python -m venv .venv
# Activate the environment for your platform.
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python export_kinematics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python export_actor.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_release.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate.py
```

No retraining is performed. Measured runtime versions are in `evaluation/summary.json`. Full measured episode rows and reference traces are distributed with the model and Kaggle dataset.

## Protocol and interpretation

100 prespecified targets: 20 each for segment, quadratic/cubic Bezier, sinusoid and arc. The same cases are paired across frozen SAC and fixed DLS (damping 0.02, gain 0.8), with position/joint sensor-noise standard deviations 0, 0.002 and 0.005 in metres/radians. This is 600 episodes, not 600 unique independent trajectories. Controller settings and cases were fixed before the run; the original training seed history cannot be reconstructed.

No-noise strict completion: SAC 93/100, DLS 90/100. Mean tracking RMS over acquired episodes, including failures: 5.21 mm and 2.38 mm. DLS is more accurate in this experiment; the observed completion difference does not establish superiority. Per-condition Wilson intervals are descriptive, not training-seed variance or a formal stratified-sampling inference.

The original `completed` flag can be true on a stagnation stop and can terminate immediately after advancing the cursor before the last point is within tolerance. The original dynamics are retained; strict completion additionally verifies final cursor, tracking phase, actual endpoint distance below 25 mm and no tracking error above 60 mm. Tests preserve a regression example of the flag issue.

These are position-only default kinematics, not a calibrated physical robot. No dynamics, contact, orientation, collision or real-time safety validation is included. Source and weights retain Dev Kumar's MIT notice; robot-description terms and provenance are retained separately.
