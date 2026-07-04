# UR7e Target Reaching & Obstacle Avoidance

Controlling a Universal Robots **UR7e** cobot (6 axes) via reinforcement learning in PyBullet simulation, exploring **two independent approaches** side by side:

- **[`UR7e-Stablebaseline3Approach-TargetReaching-ObstacleAvoidance/`](UR7e-Stablebaseline3Approach-TargetReaching-ObstacleAvoidance)** — environments written from scratch, joint-velocity control, under **Stable-Baselines3 / PyTorch**. Covers reaching, SAC vs. PPO comparison, HER, curriculum learning, avoidance, and a hybrid IK+RL approach.
- **[`UR7e-TensofFlowApproach-ObstacleAvoidance/`](UR7e-TensofFlowApproach-ObstacleAvoidance)** — a 3-phase architecture with **transfer learning**, Cartesian pose control + inverse kinematics + potential fields, under **tf-agents / TensorFlow**.

Each subfolder has its **own README** detailing the philosophy and files; this document only explains how to get started.

This batch specifically demonstrates breadth across RL methodology: algorithm comparison with statistical testing (SAC vs. PPO), sparse-reward techniques (HER), curriculum learning, reward shaping, and a hybrid classical-control/RL pipeline — as well as two structurally different control paradigms (joint-velocity vs. Cartesian+IK).

---

## 1. Essential prerequisite: the `ur7e_pybullet/` folder

**No script works without it, and it is not included in this repository.**

`ur7e_pybullet/` contains the robot model (`ur7e.urdf` + `meshes/` folder). All environments load it via a **relative path**: on launch, the script changes into the URDF's folder (`os.chdir`) since mesh paths are relative to it.

**To do**: copy `ur7e_pybullet/` **into the task subfolder** you're launching from (or at the same level, depending on what the script expects). The simplest approach is to keep a copy next to each group of scripts you use. See [`MISSING_PREREQUISITE.md`](MISSING_PREREQUISITE.md) for the full recovery procedure.

> Note: this folder is actually the complete `ur_description` package from Universal Robots. There is no dedicated "ur7e" mesh; the URDF reuses UR5e geometry (same kinematics, ~850 mm reach).

---

## 2. Installation — two separate environments

The two approaches rely on **incompatible** libraries (tf-agents vs. stable-baselines3). Two separate virtual environments are required.

### For approach 1 (from-scratch engines, SB3)
```bash
python3 -m venv venv_sb3
source venv_sb3/bin/activate
pip install numpy scipy matplotlib pybullet tqdm stable-baselines3 gymnasium torch
```

### For approach 3 (3-phase tf-agents)
```bash
python3 -m venv venv_fred
source venv_fred/bin/activate
pip install numpy scipy matplotlib pybullet tqdm "tensorflow==2.15.*" "tf-agents==0.19.*"
```

See `requirements.txt` for details.

---

## 3. Quick start

Start with `h_outils_communs/check_ur7e_joints.py` (approach 1) to confirm the robot model loads and the joint indices are as expected, then follow the subfolder README matching the task you're interested in — the SAC vs. PPO comparison (`UR7e-TargetReaching-180-ComparaisonPPOvsSAC/`) is the best entry point for the methodology used throughout this batch.

## License

MIT — see the root [LICENSE](../LICENSE).
