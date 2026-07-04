# a — Reaching 180° + rigorous SAC vs PPO comparison

## Philosophy

The **methodological** core of the project. Rather than just training an agent, this folder **rigorously compares** two reference continuous-control RL algorithms, SAC (off-policy) and PPO (on-policy), under strictly identical conditions, following genuine scientific practice (multiple seeds, a shared test set, a statistical test). The supporting task is deliberately simple: bring the end-effector to a target in the half-space in front of the robot ("180°").

## Comparison principle

- **Identical conditions for both**: same number of steps, same seeds, same test-target set (generated once, never seen during training).
- **Multiple seeds** per algorithm → measuring variability, not a lucky run.
- **Separation of collection / analysis / plotting**: one script produces the data, another does the statistics, a third draws the figures. None does two jobs at once.

## Files and order of use

| File | Role |
|---|---|
| `bullet_ur7e.py` | UR7e physics engine (velocity control, reaching, reward). |
| `ur7e_wrapper.py` | Gymnasium bridge for Stable-Baselines3. |
| `train_sac_ur7e.py` | Trains a SAC agent. |
| `train_ppo_ur7e.py` | Trains a PPO agent. |
| `run_experiment.py` | **Runs the whole experiment**: N seeds × 2 algorithms, evaluates on the shared test set, writes raw CSVs. Plots nothing. |
| `analyze_results.py` | Statistics (mean ± std) + **Mann-Whitney U test**. |
| `plot_results.py` | Figures: learning curves, box plots, cost/performance scatter. |
| `success_multi_seuils.py` | Recomputes success rate at several thresholds (5 mm / 2 cm / 5 cm) from already-recorded distances — no retraining needed. |
| `evaluate_ur7e.py` | Visual demonstration of a trained agent + metrics. |

## How to run

```bash
# (copy ur7e_pybullet/ here first)
caffeinate -i python run_experiment.py     # produces resultats_xp/*.csv
python analyze_results.py                  # reads resultats_xp/, outputs stats
python plot_results.py                     # reads resultats_xp/, outputs figures
python success_multi_seuils.py             # multi-threshold success rate
```

`run_experiment.py` has two parameters at the top of the file: `N_RUNS` (seeds per algorithm) and `N_STEPS` (steps per run). Reduce them for a quick test.

## Reference result

At 100,000 steps, 5 seeds, 5 mm threshold: SAC reaches ~13% success vs. ~1% for PPO (statistically significant difference, p ≈ 0.01), with SAC converging ~8x faster than PPO. Conclusion: under a limited interaction budget, SAC is clearly superior.
