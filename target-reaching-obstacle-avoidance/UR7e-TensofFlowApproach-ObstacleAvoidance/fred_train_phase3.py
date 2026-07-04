# ==============================================================================
# FICHIER : fred_train_phase3.py
# RÔLE : Lance l'entraînement de la PHASE 3 (départ fixe, cible aléatoire gauche).
#        TRANSFER depuis la phase 2 (checkpoint fred_policy_phase2_ckpt).
#        Produit fred_policy_phase3_final = MODÈLE FINAL.
#
# Usage : caffeinate -i python fred_train_phase3.py
#   (nécessite d'avoir terminé la phase 2 avant)
#
# ---- EXPÉRIENCE IK (optimisation vitesse) --------------------------------
# IK_MAX_ITER contrôle le coût/précision du solveur IK appelé à chaque pas.
#   - 20 (défaut PyBullet) : précis, lent  -> référence
#   - 5-8 : plus rapide, IK moins précise   -> à comparer
# Protocole de comparaison (isolation de variable) :
#   1) lancer un run court (NUM_ITERATIONS=10_000) avec IK_MAX_ITER=20,
#      noter le débit (it/s) ET la courbe success_rate du CSV.
#   2) relancer à l'identique avec IK_MAX_ITER=6.
#   3) comparer : le débit doit monter ; vérifier que success_rate ne s'effondre
#      PAS. Le vrai juge = temps pour atteindre un succès donné, pas l'it/s brut.
# IMPORTANT : garder la MÊME valeur pour l'entraînement ET la visualisation.
# --------------------------------------------------------------------------
IK_MAX_ITER = 20          # <-- change ici pour la comparaison (ex: 6)
IK_RESIDUAL = 1e-4

from fred_phase3_env import FredPhase3Env
from fred_train_phase import train_phase

NUM_ITERATIONS = 100_000

if __name__ == "__main__":
    train_phase(
        make_env=lambda seed: FredPhase3Env(
            use_gui=False, seed=seed,
            ik_max_iter=IK_MAX_ITER, ik_residual=IK_RESIDUAL),
        num_iterations=NUM_ITERATIONS,
        csv_path="metrics_phase3.csv",
        policy_save_dir="fred_policy_phase3",
        load_from="fred_policy_phase2",   # reprend la phase 2
        seed=0)
