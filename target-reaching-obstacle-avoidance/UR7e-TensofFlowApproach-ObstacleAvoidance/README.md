# 3 — Three phases with transfer learning (tf-agents)

## Philosophy of this approach

A **radically different** approach from the from-scratch engines (folder 1), on two fundamental points.

**1. The control scheme.** Here the agent does not drive joints in velocity: it commands a **Cartesian end-effector displacement** (Δpose), and **inverse kinematics** (PyBullet IK) translates that displacement into joint angles. RL therefore doesn't need to learn arm kinematics — only *where* to go. The observation is based on **potential fields**: an attractive vector toward the target and repulsive vectors away from obstacles, computed at 3 control points (end-effector `tool0`, wrist `wrist_1`, forearm `forearm`).

**2. Progressive learning via transfer learning.** Rather than training a complex task all at once, it's learned in **3 chained phases**, each phase resuming the weights of the previous one:

- **Phase 1** — simple reaching: start = rest pose, random target, no obstacle. The agent learns to reach.
- **Phase 2** — avoidance with **fixed start AND end** on either side of a fixed box (15x27x22 cm) blocking the depth axis. Goes around from above (21 cm margin under the volume's ceiling). The agent learns to avoid.
- **Phase 3** — fixed start, **random target** in the outer-left half (with rejection sampling to guarantee reachability). The agent generalizes avoidance to varied targets.

The key enabler of transfer: the **observation has a constant dimension of 15** across all 3 phases (repulsive terms are simply zero in phase 1 with no obstacle). The networks thus have the same shape from one phase to the next, and weights can be resumed directly via a tf-agents `Checkpointer`.

## Files

### Core (environments)
| File | Role |
|---|---|
| `fred_potential_fields.py` | Attractive/repulsive potential fields (bounded Khatib formula). Shared dependency. |
| `fred_phase_base.py` | **Base environment** shared by all 3 phases: IK, potential fields, 15D observation, full-arm collision. Workspace volume defined here. |
| `fred_phase1_env.py` | Phase 1 (reaching, no obstacle). |
| `fred_phase2_env.py` | Phase 2 (fixed A→B, fixed box). |
| `fred_phase3_env.py` | Phase 3 (fixed start, random target in outer half + rejection sampling). |

### Training
| File | Role |
|---|---|
| `fred_train_phase.py` | Generic SAC training + **transfer learning** via `Checkpointer`. Two trackers (stochastic rolling + deterministic eval every 1000 steps). Called by the three scripts below. |
| `fred_train_phase1.py` | Runs phase 1 (cold start). Produces `fred_policy_phase1_final` + `_ckpt`. |
| `fred_train_phase2.py` | Runs phase 2 (transfer from the phase-1 checkpoint). |
| `fred_train_phase3.py` | Runs phase 3 (transfer from the phase-2 checkpoint) = final model. |

### Metrics and visualization
| File | Role |
|---|---|
| `fred_metrics.py` | Rolling-metrics tracker (success/collision/timeout, distances, percentiles) + CSV output. |
| `fred_visualize_phase.py` | Visualizes a phase's policy: `python fred_visualize_phase.py 1|2|3`. B-spline smoothing + collision detection. |
| `fred_show_env.py` | **Static** visualization of a phase's environment (robot at rest + obstacle + start/end markers), no motion. Use before training to verify geometry. |
| `fred_eval_gui.py` | GUI evaluation/visualization of a policy. |

## How to run

```bash
# (copy ur7e_pybullet/ here first, and activate the tf-agents venv)

# 1. Verify each phase's geometry (no motion)
python fred_show_env.py 1
python fred_show_env.py 2
python fred_show_env.py 3

# 2. Train the 3 phases SEQUENTIALLY (each resumes the previous one)
caffeinate -i python fred_train_phase1.py
caffeinate -i python fred_train_phase2.py
caffeinate -i python fred_train_phase3.py

# 3. Visualize the final policy
python fred_visualize_phase.py 3
```

## Points of attention

- **Phase order is mandatory**: phase 2 needs the phase-1 checkpoint, phase 3 needs the phase-2 checkpoint. Run them in order.
- **`caffeinate -i`** before every training run (prevents the Mac from sleeping, which would skew timings and could interrupt training).
- **Speed**: this tf-agents approach is significantly slower than SB3 (~14 it/s vs. ~60 fps), since the dominant cost is TensorFlow training on CPU, not the simulation itself. This is a known limitation of the approach.
- **Workspace volume**: `x in [-0.42, 0.42]`, `y in [0.25, 0.52]`, `z in [0.15, 0.58]` m, defined in `fred_phase_base.py` — calibrated to stay within the UR7e's reach.
