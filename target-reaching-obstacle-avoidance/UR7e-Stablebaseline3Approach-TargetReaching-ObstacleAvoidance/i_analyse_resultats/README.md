# i — Results analysis and figures

## Philosophy

**Post-hoc analysis** scripts: they train nothing, they **read an already-trained model** (or result CSVs) and produce the report's analysis figures — spatial maps, heatmaps, capability studies. The goal is to understand **where** and **why** the agent succeeds or fails, not just its overall rate.

## Files

| File | Produces |
|---|---|
| `analyse_250k.py` | Capability study of a SAC agent trained for 250k steps (single agent, no multi-seed averaging): mapping, 3D trajectories, distance/time failure analysis, multi-threshold rates. |
| `analyse_spatiale.py` | 2D projections (side view x-z, top view x-y) + success vs. radial distance, at 5 mm and 5 cm thresholds. |
| `carte_succes.py` | Grid heatmap (x-z plane) of success rate by zone, at 5 mm and 5 cm thresholds. |
| `heatmap_fine.py` | Fine-resolution performance heatmaps. |
| `density_3d_volume.py` | 3D performance density across the workspace volume. |
| `density_3d_volume_360.py` | Same, over the full 360° workspace. |

## Important prerequisite

These scripts expect a **trained model** (e.g. `sac_ur7e_reach.zip` at 250k steps) and sometimes the `resultats_xp/` CSVs. Since models are **not included** in this repository, the corresponding task (folder `a` or `b`) must be retrained first to regenerate these files, then the model path at the top of each script adjusted if needed.

## How to run (example)

```bash
python analyse_250k.py        # requires the trained 250k model
python carte_succes.py        # requires a model + target set
```
