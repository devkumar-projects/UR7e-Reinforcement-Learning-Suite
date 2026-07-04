# b — Reaching on the full workspace (360°)

## Philosophy

After the half-space comparison, this moves to the **full workspace** around the base (360°), which is significantly harder: the agent must succeed regardless of the target's direction. Two variants coexist here:

- **standard**: starts from random poses, simple reward.
- **shaped**: **shaped** reward (bonus when getting within 3 cm, speed penalty on approach) and starting from a **fixed rest pose**. This is the recipe that produced the best model (~98-99% at 5 cm).

The core idea behind "shaped": guide learning by rewarding good approach behavior (slowing down near the target), rather than letting the agent discover everything on its own.

## Files

| File | Role |
|---|---|
| `bullet_ur7e_360.py` | Baseline full-workspace engine (random starts). |
| `ur7e_wrapper_360.py` | Associated Gymnasium bridge. |
| `train_sac_ur7e_360.py` | Standard 360° SAC training. |
| `bullet_ur7e_360_shaped.py` | 360° engine with reward shaping + fixed start. |
| `ur7e_wrapper_360_shaped.py` | Associated Gymnasium bridge (shaped version). |
| `train_sac_360_shaped.py` | Trains the "shaped" recipe (the best one). |
| `visualize_360_shaped.py` | GUI visualization of the shaped model over ~50 targets. |
| `plot_approach_curves_shaped.py` | Effector-to-target distance vs. step curves, success vs. failure (shows the approach "braking"). |

## How to run (shaped variant, recommended)

```bash
caffeinate -i python train_sac_360_shaped.py    # trains the shaped model
python visualize_360_shaped.py                  # observe the behavior
python plot_approach_curves_shaped.py           # approach-profile figure
```
