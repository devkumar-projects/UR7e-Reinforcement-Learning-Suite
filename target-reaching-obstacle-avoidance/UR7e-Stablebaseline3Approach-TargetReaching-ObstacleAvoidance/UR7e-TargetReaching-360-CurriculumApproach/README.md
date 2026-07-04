# d — Adaptive curriculum

## Philosophy

Instead of imposing the demanding success threshold from the start, the agent learns **in stages of increasing difficulty**. It starts with a wide threshold (easy to reach, hence frequent learning signal), and **as soon as it masters a stage**, the threshold tightens automatically. The idea: avoid getting stuck on a task that's initially too hard, by building up competence progressively.

This is an alternative to reward shaping (folder b) for solving the same problem — learning precision over the full workspace.

## Files

| File | Role |
|---|---|
| `train_sac_curriculum_360.py` | SAC training with adaptive curriculum on the success threshold. |
| `continue_train_curriculum_360.py` | Resumes training by reloading the model, the replay buffer, **and the curriculum state** (current stage) — no dip on restart. |
| `visualize_curriculum_360.py` | GUI visualization of the model, to observe the "fine braking" on approach. |

## How to run

```bash
caffeinate -i python train_sac_curriculum_360.py
# (if interrupted:) caffeinate -i python continue_train_curriculum_360.py
python visualize_curriculum_360.py
```

## Note

This folder reuses the 360° engine and wrapper from `UR7e-TargetReaching-360/`. If running these scripts from this folder, make sure the imported modules (360° engine/wrapper) are accessible — either by copying the needed files here, or by launching from a folder where they are present.
