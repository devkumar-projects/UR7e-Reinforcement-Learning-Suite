# h — Common tools

## Philosophy

Small cross-cutting utilities, independent of any specific task. Guiding principle: **assume nothing, verify everything**. The UR7e was derived from the UR30; rather than assuming the joint structure is identical, it is explicitly inspected.

## Files

| File | Role |
|---|---|
| `check_ur7e_joints.py` | Verifies the indices of the controllable (REVOLUTE) joints and the end-effector (`tool0`) of the UR7e, by loading the URDF and listing each joint. Run once to confirm the structure. |

## How to run

```bash
# (copy ur7e_pybullet/ here first)
python check_ur7e_joints.py
```

Expected result: 12 joints total, 6 REVOLUTE at indices 2 to 7, end-effector `tool0` at index 10.
