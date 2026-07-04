# f — Hybrid IK → RL → IK architecture

## Philosophy

An **engineering** idea: don't ask RL to do everything, only the part it does better than IK. The pipeline splits the trajectory into three:

1. **IK** brings the end-effector to an **entry** point (simple zone).
2. **RL** takes over for the hard part: **going around the obstacle** (a ~1 m x Ø50 cm cylinder, simplified human geometry) and reaching an **exit** point.
3. **IK** finishes from the exit point to the final target.

RL is thus only responsible for the avoidance maneuver (where IK fails), and IK for the trivial portions (where it's optimal). A pragmatic compromise between learning and classical computation.

## Files

| File | Role |
|---|---|
| `bullet_ur7e_hybrid.py` | RL-phase engine: going around the cylinder, entry point → exit point. |
| `ur7e_wrapper_hybrid.py` | Gymnasium bridge, 28D observation. |
| `train_sac_hybrid.py` | SAC training from scratch on the RL phase (avoidance). |
| `continue_train_hybrid.py` | Resumes training (model + buffer). |

## How to run

```bash
caffeinate -i python train_sac_hybrid.py
# (if interrupted:) caffeinate -i python continue_train_hybrid.py
```
