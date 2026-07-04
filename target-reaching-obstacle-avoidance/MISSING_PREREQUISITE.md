# Missing prerequisite — `ur7e_pybullet/`

This Target Reaching / Obstacle Avoidance batch uses PyBullet and expects a local folder:

```text
ur7e_pybullet/
├── ur7e.urdf
└── meshes/
```

This folder is not included in the final deliverable. The scripts are kept for analysis and future reuse, but they cannot be run immediately on a fresh machine without this robot model.

Recommended recovery procedure:

1. Get the UR model from the Universal Robots / `ur_description` package matching the version the team used.
2. Recreate a `ur7e_pybullet/` folder containing the URDF and meshes expected by the scripts.
3. Place this folder next to the task scripts being run (e.g. next to `bullet_ur7e.py`).
4. Run `h_outils_communs/check_ur7e_joints.py` first to verify joint ordering and model consistency.

Do not modify the reward or training scripts before validating that the robot loads correctly in PyBullet.
