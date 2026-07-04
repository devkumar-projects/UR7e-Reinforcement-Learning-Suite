# c — HER (Hindsight Experience Replay)

## Philosophy

On the 360° workspace with random starts, the reward is **sparse**: the agent very rarely reaches the target by chance early on, so it learns slowly. **HER** addresses this: it **rewrites failed episodes** as if they had targeted wherever the agent actually ended up. Every failure thus becomes a useful learning example for "reaching that point" — massively densifying the learning signal.

The target here is deliberately a **5 cm zone** around the goal (not a millimeter): close enough that an IK servo loop can take over afterward.

## Included debugging process

HER wasn't "taking off" in the 360° + random-starts setting. The `test_her_*` files are test benches to **isolate the cause** on an easy canonical case, before returning to the full problem.

## Files

| File | Role |
|---|---|
| `ur7e_wrapper_her.py` | **Goal-conditioned** wrapper (required for HER: separate observation + goal). |
| `train_sac_her_360.py` | SAC + HER training, 5 cm zone target. |
| `continue_train_her_360.py` | Resumes HER training without starting from scratch. |
| `test_her_facile.py` / `test_her_entfix.py` / `test_her_h50.py` | Test benches to diagnose HER convergence (easy case, entropy fix, horizon 50). |
| `density_heatmap_slices_her.py` | Density heatmaps by slice (XY planes swept in z) of the HER model. |

## How to run

```bash
caffeinate -i python train_sac_her_360.py        # HER training
# (if interrupted:) caffeinate -i python continue_train_her_360.py
python density_heatmap_slices_her.py             # visualize results
```
