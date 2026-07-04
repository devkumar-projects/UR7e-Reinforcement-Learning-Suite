# ==============================================================================
# FICHIER : fred_train_phase1.py
# RÔLE : Lance l'entraînement de la PHASE 1 (reaching sans obstacle).
#        Démarrage à froid (pas de transfer). Produit fred_policy_phase1_final.
#
# Usage : caffeinate -i python fred_train_phase1.py
# ==============================================================================

from fred_phase1_env import FredPhase1Env
from fred_train_phase import train_phase

NUM_ITERATIONS = 60_000

if __name__ == "__main__":
    train_phase(
        make_env=lambda seed: FredPhase1Env(use_gui=False, seed=seed),
        num_iterations=NUM_ITERATIONS,
        csv_path="metrics_phase1.csv",
        policy_save_dir="fred_policy_phase1",
        load_from=None,                 # phase 1 = démarrage à froid
        seed=0)
