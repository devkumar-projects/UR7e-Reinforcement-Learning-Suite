# ==============================================================================
# FICHIER : fred_metrics.py
# RÔLE : Suivi des métriques d'entraînement (glissantes sur 100 épisodes) +
#        enregistrement CSV pour analyse statistique a posteriori.
#        Partagé par les 3 phases (transfer learning UR7e / Fred).
#
# Métriques glissantes (fenêtre 100 épisodes) :
#   - success_rate, collision_rate, timeout_rate
#   - mean_distance_cm, distance_std_cm, distance_median_cm, distance_p90_cm
#   - ep_len_mean, ep_len_std
#   - ep_rew_mean (= ep_mean_raw)
#   - eval_success_rate (déterministe, lissé sur les dernières évals)
# Colonnes temporelles : timesteps, wall_time_s
# ==============================================================================

import csv
import os
import time
from collections import deque
import numpy as np


SUCCESS_REWARD_THRESHOLD = 50.0
COLLISION_REWARD_THRESHOLD = -50.0


class MetricsTracker:
    """Trackers glissants (100 épisodes) + écriture CSV."""

    def __init__(self, csv_path, window=100, eval_window=10):
        self.window = window
        self.csv_path = csv_path
        self.start_time = time.time()

        # fenêtres glissantes (politique de collecte, stochastique)
        self.successes = deque(maxlen=window)
        self.collisions = deque(maxlen=window)
        self.timeouts = deque(maxlen=window)
        self.distances_cm = deque(maxlen=window)
        self.ep_lens = deque(maxlen=window)
        self.ep_rews = deque(maxlen=window)
        self.episodes = 0

        # eval déterministe : lissée sur les dernières évaluations
        self.eval_successes = deque(maxlen=eval_window)

        # buffers internes par épisode en cours (par env, ici 1 seul)
        self._cur_len = 0
        self._cur_rew = 0.0

        # init CSV avec en-tête
        self._init_csv()

    HEADERS = [
        "timesteps", "wall_time_s", "episodes",
        "success_rate", "collision_rate", "timeout_rate",
        "mean_distance_cm", "distance_std_cm", "distance_median_cm",
        "distance_p90_cm", "ep_len_mean", "ep_len_std", "ep_rew_mean",
        "eval_success_rate",
    ]

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADERS)

    def record_step(self, reward):
        """Accumule un pas (reward) dans l'épisode en cours."""
        self._cur_len += 1
        self._cur_rew += float(reward)

    def end_episode(self, terminal_reward, final_distance_cm):
        """Clôt un épisode : classe succès/collision/timeout et met à jour."""
        r = float(terminal_reward)
        is_succ = 1.0 if r >= SUCCESS_REWARD_THRESHOLD else 0.0
        is_coll = 1.0 if r <= COLLISION_REWARD_THRESHOLD else 0.0
        is_timeout = 1.0 if (is_succ == 0.0 and is_coll == 0.0) else 0.0
        self.successes.append(is_succ)
        self.collisions.append(is_coll)
        self.timeouts.append(is_timeout)
        self.distances_cm.append(float(final_distance_cm))
        self.ep_lens.append(self._cur_len)
        self.ep_rews.append(self._cur_rew)
        self.episodes += 1
        self._cur_len = 0
        self._cur_rew = 0.0

    def update_eval(self, eval_success_rate):
        """Ajoute une évaluation déterministe (lissée en glissant)."""
        self.eval_successes.append(float(eval_success_rate))

    # ---- accès aux valeurs courantes (pour le moniteur tqdm) ----
    @staticmethod
    def _safe_mean(d):
        return float(np.mean(d)) if len(d) else 0.0

    @property
    def success_rate(self):
        return self._safe_mean(self.successes)

    @property
    def collision_rate(self):
        return self._safe_mean(self.collisions)

    @property
    def timeout_rate(self):
        return self._safe_mean(self.timeouts)

    @property
    def mean_distance_cm(self):
        return self._safe_mean(self.distances_cm)

    @property
    def eval_success_rate(self):
        return self._safe_mean(self.eval_successes)

    def snapshot(self, timesteps):
        """Calcule toutes les métriques et les écrit dans le CSV."""
        dist = np.array(self.distances_cm) if self.distances_cm else np.array([0.0])
        lens = np.array(self.ep_lens) if self.ep_lens else np.array([0.0])
        row = {
            "timesteps": timesteps,
            "wall_time_s": round(time.time() - self.start_time, 1),
            "episodes": self.episodes,
            "success_rate": round(self.success_rate, 4),
            "collision_rate": round(self.collision_rate, 4),
            "timeout_rate": round(self.timeout_rate, 4),
            "mean_distance_cm": round(float(np.mean(dist)), 2),
            "distance_std_cm": round(float(np.std(dist)), 2),
            "distance_median_cm": round(float(np.median(dist)), 2),
            "distance_p90_cm": round(float(np.percentile(dist, 90)), 2),
            "ep_len_mean": round(float(np.mean(lens)), 1),
            "ep_len_std": round(float(np.std(lens)), 1),
            "ep_rew_mean": round(self._safe_mean(self.ep_rews), 3),
            "eval_success_rate": round(self.eval_success_rate, 4),
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row[h] for h in self.HEADERS])
        return row
