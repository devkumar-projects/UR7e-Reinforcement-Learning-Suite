# g — Obstacle crossing

## Philosophy

A variant of the avoidance task, but with a **deliberately difficult starting condition**: the agent starts from a zone **right against the obstacle** (the "start" side, nearly in collision) and must reach the clear zone on the other side, never hitting the cylinder, while keeping a margin ≥ 5 cm. This is the most constrained case in the series: there is no margin for error at the start.

This folder also includes a **diagnostic** script (`diag_crossing.py`), born from a concrete problem: episodes sometimes ended immediately (start already in collision). The diagnostic prints the termination reason and the initial state.

## Files

| File | Role |
|---|---|
| `bullet_ur7e_crossing.py` | Crossing engine (start zone against obstacle → clear zone). |
| `ur7e_wrapper_crossing.py` | Gymnasium bridge, 24D observation. |
| `train_sac_crossing.py` | SAC training from scratch. |
| `continue_train_crossing.py` | Resumes training (model + buffer). |
| `diag_crossing.py` | Diagnostic: why do episodes end early? Prints done_reason + state at step 1. |

## How to run

```bash
caffeinate -i python train_sac_crossing.py
python diag_crossing.py            # if episodes end too quickly
# (if interrupted:) caffeinate -i python continue_train_crossing.py
```
