# ==============================================================================
# FICHIER : fred_train_phase2.py
# RÔLE : Lance l'entraînement de la PHASE 2 (A->B fixes, évitement obstacle).
#        TRANSFER depuis la phase 1 (checkpoint fred_policy_phase1_ckpt).
#        Produit fred_policy_phase2_final.
#
# Usage : caffeinate -i python fred_train_phase2.py
#   (nécessite d'avoir terminé la phase 1 avant)
# ==============================================================================

from fred_phase2_env import FredPhase2Env
from fred_train_phase import train_phase

NUM_ITERATIONS = 10_000

if __name__ == "__main__":
    train_phase(
        make_env=lambda seed: FredPhase2Env(use_gui=False, seed=seed),
        num_iterations=NUM_ITERATIONS,
        csv_path="metrics_phase2.csv",
        policy_save_dir="fred_policy_phase2",
        load_from="fred_policy_phase2",   # reprend la phase 1
        seed=0)
