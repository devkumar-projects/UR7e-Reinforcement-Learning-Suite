# ==============================================================================
# FICHIER : run_experiment.py
# RÔLE : Collecte de données pour la comparaison scientifique SAC vs PPO.
#
# Entraîne chaque algorithme sur N_RUNS graines aléatoires indépendantes, puis
# évalue chaque modèle figé sur un JEU DE CIBLES DE TEST COMMUN (mêmes cibles
# pour tous, généré une seule fois). Toutes les données brutes sont écrites dans
# resultats_xp/ pour analyse ultérieure (analyze_results.py, plot_results.py).
#
# Ce script NE TRACE RIEN et NE DÉCIDE RIEN : il produit des données.
#
# --- PARAMÈTRES MODIFIABLES -------------------------------------------------
#   N_RUNS  : nombre de graines par algorithme
#   N_STEPS : nombre d'étapes d'entraînement par run
# ----------------------------------------------------------------------------
#
# Reproductibilité : chaque run fixe la graine (numpy, torch, env) -> rejouable.
# Coût indicatif : 5 runs x 2 algos ~ 4 h sur CPU portable.
# ==============================================================================

import os
import time
import csv
import json
import numpy as np
import torch

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from ur7e_wrapper import UR7eReachEnv

# ============================= PARAMÈTRES ====================================
N_RUNS = 5            # nombre de graines par algorithme
N_STEPS = 100_000     # étapes d'entraînement par run
N_TEST_TARGETS = 50   # nombre de cibles du jeu de test commun
MAX_EVAL_STEPS = 300  # pas max par épisode d'évaluation
SUCCESS_THRESHOLD = 0.005   # 5 mm
TEST_SEED = 12345     # graine dédiée au jeu de test (fixe, jamais utilisée en entraînement)

OUT_DIR = "resultats_xp"
ALGOS = ["sac", "ppo"]
# =============================================================================


def set_global_seeds(seed):
    """Fixe toutes les sources d'aléa pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_test_targets():
    """
    Génère un jeu de cibles de test FIXE et COMMUN à tous les modèles.
    Généré une seule fois avec TEST_SEED, sauvegardé sur disque pour garantir
    que tous les runs (et toute ré-exécution) utilisent exactement les mêmes cibles.
    """
    path = os.path.join(OUT_DIR, "test_targets.npy")
    if os.path.exists(path):
        return np.load(path)

    rng = np.random.RandomState(TEST_SEED)
    targets = []
    for _ in range(N_TEST_TARGETS):
        x = rng.uniform(0.25, 0.65)
        y = rng.uniform(-0.45, 0.45)
        z = rng.uniform(0.15, 0.70)
        targets.append([x, y, z])
    targets = np.array(targets)
    np.save(path, targets)
    return targets


def evaluate_model(model, test_targets):
    """
    Évalue un modèle figé sur le jeu de cibles commun.
    Retourne une liste de dicts (une entrée par cible) avec les métriques.
    """
    env = UR7eReachEnv(render_mode=None, max_episode_len=MAX_EVAL_STEPS,
                       success_threshold=SUCCESS_THRESHOLD)
    results = []

    for tgt in test_targets:
        obs, _ = env.reset()
        # Force la cible de test (on écrase la cible aléatoire du reset)
        env.engine.target = np.array(tgt, dtype=np.float64)
        env.engine._spawn_target_marker()
        obs = env.engine._get_observation()

        done = False
        steps = 0
        energy = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            energy += float(np.linalg.norm(action))
            done = terminated or truncated
            steps += 1

        dist = info["distance"]
        results.append({
            "distance_m": dist,
            "success": int(dist < SUCCESS_THRESHOLD),
            "steps": steps,
            "energy": energy,
        })

    env.close()
    return results


def train_one_run(algo, seed):
    """Entraîne un agent (algo, seed), le sauvegarde, et l'évalue. Retourne un résumé."""
    print(f"\n{'='*60}\n[{algo.upper()}] graine {seed} — entraînement {N_STEPS:,} étapes\n{'='*60}")
    set_global_seeds(seed)

    log_dir = os.path.join(OUT_DIR, f"{algo}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)

    env = UR7eReachEnv(render_mode=None, max_episode_len=MAX_EVAL_STEPS,
                       success_threshold=SUCCESS_THRESHOLD)
    env = Monitor(env, log_dir)  # écrit monitor.csv (courbe d'apprentissage)
    env.reset(seed=seed)

    if algo == "sac":
        model = SAC("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                    learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                    gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
                    learning_starts=1000)
    else:
        model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    gamma=0.99, gae_lambda=0.95)

    t0 = time.time()
    model.learn(total_timesteps=N_STEPS, progress_bar=True)
    train_time = time.time() - t0

    model_path = os.path.join(OUT_DIR, f"{algo}_seed{seed}")
    model.save(model_path)
    env.close()

    # Évaluation sur le jeu de test commun
    print(f"[{algo.upper()}] graine {seed} — évaluation sur {N_TEST_TARGETS} cibles...")
    eval_results = evaluate_model(model, TEST_TARGETS)

    # Sauvegarde des résultats d'éval détaillés (par cible)
    eval_csv = os.path.join(OUT_DIR, f"eval_{algo}_seed{seed}.csv")
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["distance_m", "success", "steps", "energy"])
        w.writeheader()
        w.writerows(eval_results)

    # Résumé agrégé de ce run
    dists = np.array([r["distance_m"] for r in eval_results])
    succ = np.array([r["success"] for r in eval_results])
    steps = np.array([r["steps"] for r in eval_results])
    energy = np.array([r["energy"] for r in eval_results])

    summary = {
        "algo": algo,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "success_rate": float(succ.mean()),
        "dist_median_mm": float(np.median(dists) * 1000),
        "dist_mean_mm": float(dists.mean() * 1000),
        "dist_min_mm": float(dists.min() * 1000),
        "dist_max_mm": float(dists.max() * 1000),
        "steps_mean": float(steps.mean()),
        "energy_mean": float(energy.mean()),
    }
    print(f"[{algo.upper()}] graine {seed} — succès {summary['success_rate']*100:.0f}% "
          f"| médiane {summary['dist_median_mm']:.1f} mm | {train_time:.0f}s")
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    global TEST_TARGETS
    TEST_TARGETS = make_test_targets()
    print(f"Jeu de test commun : {len(TEST_TARGETS)} cibles fixes (graine {TEST_SEED}).")

    all_summaries = []
    for algo in ALGOS:
        for seed in range(N_RUNS):
            summary = train_one_run(algo, seed)
            all_summaries.append(summary)

    # Sauvegarde du tableau de synthèse global
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        w.writeheader()
        w.writerows(all_summaries)

    # Sauvegarde des métadonnées de l'expérience (traçabilité)
    meta = {
        "n_runs": N_RUNS, "n_steps": N_STEPS, "n_test_targets": N_TEST_TARGETS,
        "success_threshold_m": SUCCESS_THRESHOLD, "test_seed": TEST_SEED,
        "algos": ALGOS,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}\nExpérience terminée. Données dans '{OUT_DIR}/'.")
    print(f"  - summary.csv          : synthèse par run")
    print(f"  - eval_*_seed*.csv     : résultats détaillés par cible")
    print(f"  - */monitor.csv        : courbes d'apprentissage")
    print(f"Lance maintenant : python analyze_results.py")
    print('='*60)


if __name__ == "__main__":
    main()
