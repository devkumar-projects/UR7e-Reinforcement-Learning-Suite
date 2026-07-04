# 1 — Custom "from scratch" engines, SB3 (PyTorch)

## Philosophy of this approach

This is the **first major approach** of the project. Custom PyBullet environments are written **from scratch**, with no intermediate RL simulation framework, and trained with **Stable-Baselines3** (PyTorch).

The key design choice: the agent commands the robot **in joint velocity** (6 velocities, one per axis). There is no inverse kinematics in the loop — the agent must learn by itself how to move its joints to bring the end-effector where it wants. This is harder to learn than pose-space control (the approach in folder 3), but it is RL in its purest, strictest sense.

## Three-layer architecture (recurring across all subfolders)

Every task follows the same file pattern:

1. **`bullet_*.py`** — the **physics engine**: loads the robot, applies velocity commands, reads the state, computes the reward. This is the "body".
2. **`ur7e_wrapper_*.py`** — the **Gymnasium bridge**: wraps the engine to make it compatible with Stable-Baselines3 (observation/action spaces).
3. **`train_*.py`** / **`continue_train_*.py`** — **training** (from scratch, or resuming a model + its replay buffer).
4. **`visualize_*.py`** / **`diag_*.py`** — **visualization** and diagnostics.

## Subfolders, in logical order

| Folder | Task explored |
|---|---|
| `UR7e-TargetReaching-180-ComparaisonPPOvsSAC/` | Reaching on the half-space + rigorous **SAC vs PPO** scientific comparison (the methodological core). |
| `UR7e-TargetReaching-360/` | Reaching on the **full** workspace (360°), standard version and **shaped** version (shaped reward, ~99%). |
| `UR7e-TargetReaching-360-HERapproach/` | Exploring **HER** (Hindsight Experience Replay) to speed up learning under sparse reward. |
| `UR7e-TargetReaching-360-CurriculumApproach/` | Adaptive **curriculum**: the success threshold tightens progressively. |
| `UR7e-ObstacleAvoidance/` | Reaching + avoiding a random **spherical obstacle**. |
| `UR7e-ObstacleAvoidance-IKandRLapproach/` | **IK → RL → IK** pipeline to go around a cylinder (simplified human geometry). |
| `UR7e-ObstacleAvoidance-CloseToObstacle/` | **Crossing**: starting right against the obstacle, reaching the other side. |
| `h_outils_communs/` | Robot-structure verification utility. |
| `i_analyse_resultats/` | Analysis and figure-generation scripts (maps, heatmaps, 250k-step study). |

## Important note on cross-folder dependencies

Each subfolder is **self-contained per task**, but some analysis scripts (`i_analyse_resultats/`) and the comparison folder (`UR7e-TargetReaching-180-ComparaisonPPOvsSAC/`) assume trained models and result CSVs already exist. Since models are not included in this repository, the relevant task must be **retrained** before running its analysis. Each subfolder README states these dependencies explicitly.
