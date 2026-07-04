# ==============================================================================
# FICHIER : diag_crossing.py
# RÔLE : Diagnostiquer pourquoi les épisodes se terminent immédiatement.
#        Joue quelques épisodes, affiche le done_reason et l'état au 1er step.
# Usage : python diag_crossing.py
# ==============================================================================

import numpy as np
from bullet_ur7e_crossing import BulletUR7eCrossing

eng = BulletUR7eCrossing(gui=False, seed=0)

print("=== Diagnostic : 20 épisodes, action NULLE au 1er step ===")
reasons = {}
for i in range(20):
    eng.reset()
    # état au départ
    angles, _ = eng._get_joint_states()
    ee = eng.get_ee_position()
    d_arm = eng._arm_cylinder_distance()
    w = eng._manipulability(angles)
    within_limits = bool(np.all(angles >= eng.joint_lower) and
                         np.all(angles <= eng.joint_upper))
    # un step avec action nulle (le robot ne bouge quasiment pas)
    obs, reward, done, info = eng.step(np.zeros(6))
    reason = info.get("done_reason", "none")
    reasons[reason] = reasons.get(reason, 0) + 1
    if i < 8:
        print(f"ép {i+1:2d}: départ clearance={d_arm*100:5.1f}cm  "
              f"manip={w:.4f} (seuil {eng.manip_threshold})  "
              f"limites_ok={within_limits}  "
              f"-> 1er step: reason={reason}, reward={reward:.2f}")

print("\n=== Répartition des done_reason au 1er step (20 épisodes) ===")
for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  {r:15s} : {c}/20")

# zoom : valeurs de manipulabilité au départ sur 50 épisodes
print("\n=== Manipulabilité au départ sur 50 resets ===")
manips = []
for _ in range(50):
    eng.reset()
    angles, _ = eng._get_joint_states()
    manips.append(eng._manipulability(angles))
manips = np.array(manips)
print(f"  min={manips.min():.4f}  médiane={np.median(manips):.4f}  "
      f"max={manips.max():.4f}")
print(f"  seuil actuel = {eng.manip_threshold}")
print(f"  fraction sous le seuil = {100*np.mean(manips < eng.manip_threshold):.0f}%")

eng.close()
