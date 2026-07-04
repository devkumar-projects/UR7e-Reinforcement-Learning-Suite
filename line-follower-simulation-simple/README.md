# UR7e Trajectory Following — Reinforcement Learning (Simple Model)

Controlling a UR7e (6-DoF) cobot to follow 2D trajectories drawn on a "board", learned via Reinforcement Learning (SAC).

ENSAM Mechatronics engineering project.

---

## Package contents

```
trajectoire/
├── trajectory_generator.py    Generates random lines (5 types)
├── ur7e_traj_env.py           Gymnasium environment (approach + tracking)
├── train_traj.py              SAC training
├── viz_traj.py                RViz2 visualization (wall + line + trace)
├── visu_suivi_3d.py           3D matplotlib analysis
├── demo_suivi.launch.py       Launches the full demo at once
├── ur7e_rl.rviz                Saved RViz2 config
├── models_traj/
│   └── best_model.zip         The trained "brain" (SAC)
├── ur7e_env_v4.py             Kinematics (copied from ur7e_ws)
└── ur7e_generated.urdf        UR7e robot geometry
```

---

## Requirements

- ROS 2 Humble
- Python 3.10 + venv with: stable-baselines3, gymnasium, ikpy, numpy<2.0
- robot_state_publisher, rviz2

See `requirements.txt` for portable Python dependencies. ROS 2 dependencies are provided by the ROS installation, not by pip.

---

## Running the demo (visualization)

```bash
source /opt/ros/humble/setup.bash
cd <this-folder>
ros2 launch ./demo_suivi.launch.py
```

Everything launches at once: the robot, the trajectory-following agent, and RViz2 showing the wall (board), the target line (green), and the actual tracked path (blue).

---

## Re-training the model

```bash
cd <this-folder>
PYTHONUNBUFFERED=1 python3 -u train_traj.py
```

~5h on an RTX 4050 for 2M steps. The model is saved to `models_traj/`.

---

## Analyzing performance (3D plots)

```bash
python3 visu_suivi_3d.py
```

Generates a 3D figure: target trajectory (green) vs. actual tracking (blue).

---

## Current model performance

| Metric | Value |
|---|---|
| Line-acquisition rate | 100% |
| Completion rate | 97% |
| Mean RMS error | 5.8 mm |
| Median RMS error | 4.7 mm |

(deterministic evaluation over 100 random trajectories)

---

## Technical architecture

- **Kinematics**: ikpy on the official UR7e URDF (no sim-to-real gap)
- **Algorithm**: SAC (Soft Actor-Critic), off-policy, continuous control
- **Observation**: 29 values (6 joint angles + 3 position + progress + phase + target point + 5 lookahead points)
- **Action**: 6 joint-angle deltas
- **Phases**: APPROACH (reach the line) then TRACKING (follow it)
- **Reward**: dominant lateral accuracy + progress + anti-stagnation term

---

## ⚠ Important note: NumPy version

The `.zip` model was saved with a specific NumPy version. If loading fails (`numpy._core` not found), align the version:

```bash
pip install "numpy<2.0" --force-reinstall
```

---

## Sharing between collaborators

The 3 files that must be kept TOGETHER:
1. `models_traj/best_model.zip` — the brain
2. `ur7e_traj_env.py` + `ur7e_env_v4.py` — the environment definition
3. `ur7e_generated.urdf` — the robot geometry

The brain runs without a GPU (instant on CPU):
```python
from stable_baselines3 import SAC
model = SAC.load("models_traj/best_model")
action, _ = model.predict(observation, deterministic=True)
```

---

## License

MIT — see the root [LICENSE](../LICENSE).
