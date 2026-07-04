# e — Spherical obstacle avoidance

## Philosophy

A first step toward a task that classical inverse kinematics cannot handle natively: **avoiding an obstacle**. A spherical obstacle is placed **randomly on the segment** between the end-effector and the target, forcing the agent to go around it rather than straight through. This is where RL really earns its place: there is no simple analytical solution for the avoidance maneuver.

The observation is **extended** (28D) to include the obstacle's position and size — the agent must "see" the obstacle to avoid it.

## Files

| File | Role |
|---|---|
| `bullet_ur7e_obstacle.py` | 360° reaching engine + random spherical obstacle. |
| `ur7e_wrapper_obstacle.py` | Gymnasium bridge, observation extended to 28D (obstacle added). |
| `train_sac_obstacle.py` | SAC training from scratch (reaching + avoidance). |
| `continue_train_obstacle.py` | Resumes training (model + buffer). |
| `visualize_obstacle.py` | GUI visualization: green target, red obstacle, observe the avoidance maneuver. |

## How to run

```bash
caffeinate -i python train_sac_obstacle.py
# (if interrupted:) caffeinate -i python continue_train_obstacle.py
python visualize_obstacle.py
```
