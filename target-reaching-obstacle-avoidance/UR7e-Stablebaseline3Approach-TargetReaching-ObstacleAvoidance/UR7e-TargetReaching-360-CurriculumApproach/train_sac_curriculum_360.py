# ==============================================================================
# FICHIER : train_sac_curriculum_360.py
# RÔLE : Entraînement SAC sur le UR7e 360° avec CURRICULUM ADAPTATIF sur le
#        seuil de succès. L'agent apprend d'abord facile (seuil large) puis le
#        seuil se resserre dès qu'il maîtrise le palier courant.
#
# Paliers : 20 cm -> 15 cm -> 10 cm -> 5 cm (objectif final).
# Passage au palier suivant : dès que le success_rate glissant >= 80%.
# Budget : 400k pas (garde-fou ; un palier non maîtrisé n'est pas forcé avant
#          d'avoir eu sa chance, mais le budget total borne la durée).
#
# Récompense : DENSE -distance (+10 au succès), INCHANGÉE. Justification :
#   - le curriculum résout déjà la rareté du succès (inutile d'ajouter un bonus) ;
#   - -distance est INVARIANT au seuil -> robuste quand le seuil bouge ;
#   - ce reward a déjà fait converger proprement le 360° dense ;
#   - le +10 terminal du moteur devient automatiquement plus exigeant à chaque
#     palier (mini-bonus adaptatif gratuit).
#
# REPRISE : le buffer est sauvegardé (save_replay_buffer) + l'état du curriculum
#   (palier courant) dans un petit JSON -> reprise sans discontinuité possible
#   via continue_train_curriculum_360.py.
#
# Usage : caffeinate -i python train_sac_curriculum_360.py
# ==============================================================================

import os
import json
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ur7e_wrapper_360 import UR7eReach360Env

TOTAL_TIMESTEPS = 400_000
SEED = 0
LOG_DIR = "logs_sac_curriculum_360"
MODEL_OUT = "sac_ur7e_curriculum_360_reach"
BUFFER_OUT = "sac_ur7e_curriculum_360_buffer"
STATE_OUT = "sac_ur7e_curriculum_360_state.json"   # état du curriculum (reprise)

# Paliers du curriculum (en mètres), du plus facile au plus exigeant
CURRICULUM_THRESHOLDS = [0.20, 0.15, 0.10, 0.05]
PROMOTION_SUCCESS_RATE = 0.80      # passe au palier suivant si success_rate >= 80%
MIN_STEPS_PER_STAGE = 20_000       # garde-fou : au moins ce nombre de pas par palier
                                   # avant de pouvoir promouvoir (évite promotions hâtives
                                   # sur un success_rate calculé sur trop peu d'épisodes)


class CurriculumCallback(BaseCallback):
    """
    Resserre le seuil de succès dès que le success_rate glissant atteint le
    palier de promotion. Lit le success_rate via les épisodes du Monitor.
    Expose l'état (stage) pour la reprise.
    """

    def __init__(self, env, thresholds, promo_rate, min_steps_per_stage,
                 start_stage=0, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.thresholds = thresholds
        self.promo_rate = promo_rate
        self.min_steps_per_stage = min_steps_per_stage
        self.stage = start_stage
        self.stage_start_step = 0
        self._apply_threshold()

    def _apply_threshold(self):
        """Applique le seuil du palier courant au moteur (à travers Monitor)."""
        thr = self.thresholds[self.stage]
        # env est le Monitor ; le moteur est sous .env.engine
        base = self.env
        while hasattr(base, "env"):
            base = base.env
        base.engine.success_threshold = thr
        if self.verbose:
            print(f"\n[CURRICULUM] Palier {self.stage+1}/{len(self.thresholds)} "
                  f"-> seuil = {thr*100:.0f} cm")

    def _recent_success_rate(self, window=100):
        """success_rate sur les `window` derniers épisodes (via Monitor)."""
        # self.model.ep_info_buffer contient les infos d'épisodes récents
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) == 0:
            return 0.0
        recent = list(buf)[-window:]
        succ = [ep.get("is_success", 0.0) for ep in recent]
        return float(np.mean(succ)) if succ else 0.0

    def _recent_mean_distance(self, window=100):
        """Distance finale moyenne (m) sur les `window` derniers épisodes."""
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) == 0:
            return float("nan")
        recent = list(buf)[-window:]
        dists = [ep.get("distance") for ep in recent if ep.get("distance") is not None]
        return float(np.mean(dists)) if dists else float("nan")

    def _on_step(self):
        # vérifie la promotion à intervalle raisonnable (pas à chaque pas)
        if self.num_timesteps % 1000 != 0:
            return True
        # --- enregistre la distance moyenne atteinte dans le panneau de stats ---
        mean_dist = self._recent_mean_distance()
        # affichée en cm dans le panneau SB3 (rollout/), à côté du seuil courant
        self.logger.record("rollout/mean_distance_cm", mean_dist * 100.0)
        self.logger.record("rollout/curriculum_threshold_cm",
                           self.thresholds[self.stage] * 100.0)
        if self.stage >= len(self.thresholds) - 1:
            return True   # déjà au dernier palier (5 cm) : rien à promouvoir
        steps_in_stage = self.num_timesteps - self.stage_start_step
        if steps_in_stage < self.min_steps_per_stage:
            return True   # garde-fou : laisser le palier s'installer
        sr = self._recent_success_rate()
        if sr >= self.promo_rate:
            self.stage += 1
            self.stage_start_step = self.num_timesteps
            self._apply_threshold()
            if self.verbose:
                print(f"[CURRICULUM] Promotion à {self.num_timesteps:,} pas "
                      f"(success_rate={sr:.2f}).")
        return True

    def save_state(self, path):
        with open(path, "w") as f:
            json.dump({"stage": self.stage,
                       "stage_start_step": self.stage_start_step}, f)


class SuccessInfoWrapper:
    """Petit utilitaire pour garantir is_success dans info (pour le Monitor)."""
    pass


def make_env(seed):
    env = UR7eReach360Env(render_mode=None, max_episode_len=300,
                          success_threshold=CURRICULUM_THRESHOLDS[0],
                          random_start=True, seed=seed)
    # On enrobe step pour exposer is_success (lu par le Monitor via info_keywords)
    _orig_step = env.step

    def step_with_success(action):
        obs, reward, terminated, truncated, info = _orig_step(action)
        info = dict(info)
        info["is_success"] = 1.0 if info.get("done_reason") == "target_reached" else 0.0
        # info["distance"] est déjà fourni par le moteur (distance finale du pas).
        # Le Monitor le capturera via info_keywords -> dispo dans ep_info_buffer.
        return obs, reward, terminated, truncated, info

    env.step = step_with_success
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("1. Création de l'environnement 360° (curriculum, départ à 20 cm)...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance"))

    print("2. Création de l'agent SAC (reward dense -distance, ent_coef auto)...")
    model = SAC(
        "MlpPolicy", env,
        verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=400_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
    )

    callback = CurriculumCallback(
        env, CURRICULUM_THRESHOLDS, PROMOTION_SUCCESS_RATE,
        MIN_STEPS_PER_STAGE, start_stage=0, verbose=1,
    )

    print(f"3. Apprentissage avec curriculum ({TOTAL_TIMESTEPS:,} pas)...")
    print("   Paliers 20->15->10->5 cm, promotion si success_rate >= 80%.")
    print("   SURVEILLE [CURRICULUM] (promotions) et success_rate (dents de scie).")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                progress_bar=True)

    print("4. Sauvegarde (modèle + buffer + état curriculum)...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    callback.save_state(STATE_OUT)
    env.close()
    print(f"Terminé. Modèle '{MODEL_OUT}.zip' "
          f"(palier final atteint : {callback.stage+1}/{len(CURRICULUM_THRESHOLDS)} "
          f"= {CURRICULUM_THRESHOLDS[callback.stage]*100:.0f} cm).")
    print(f"Buffer '{BUFFER_OUT}.pkl' et état '{STATE_OUT}' sauvegardés (reprise OK).")


if __name__ == "__main__":
    main()
