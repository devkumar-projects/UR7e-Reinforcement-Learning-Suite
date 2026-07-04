# ==============================================================================
# FICHIER : continue_train_curriculum_360.py
# RÔLE : REPRENDRE l'entraînement curriculum sans discontinuité. Recharge le
#        modèle, le replay buffer ET l'état du curriculum (palier courant), puis
#        poursuit. Grâce au buffer rechargé, pas de creux au redémarrage.
#
# Usage : caffeinate -i python continue_train_curriculum_360.py
# ==============================================================================

import os
import json
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# on réutilise les briques du script d'entraînement initial
from train_sac_curriculum_360 import (
    make_env, CurriculumCallback,
    CURRICULUM_THRESHOLDS, PROMOTION_SUCCESS_RATE, MIN_STEPS_PER_STAGE,
    LOG_DIR, MODEL_OUT, BUFFER_OUT, STATE_OUT, SEED,
)

ADDITIONAL_TIMESTEPS = 200_000      # pas SUPPLÉMENTAIRES (ajustable)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not os.path.exists(MODEL_OUT + ".zip"):
        print(f"ERREUR : {MODEL_OUT}.zip introuvable. Lance d'abord l'entraînement initial.")
        return

    print("1. Recréation de l'environnement curriculum...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance"))

    print(f"2. Rechargement du modèle '{MODEL_OUT}'...")
    model = SAC.load(MODEL_OUT, env=env, device="cpu")
    print(f"   Pas déjà effectués : {model.num_timesteps:,}")

    # Rechargement du buffer (reprise sans couture, pas de creux)
    if os.path.exists(BUFFER_OUT + ".pkl"):
        print(f"   Rechargement du buffer '{BUFFER_OUT}.pkl'...")
        model.load_replay_buffer(BUFFER_OUT)
        print(f"   Buffer rechargé ({model.replay_buffer.size():,} transitions).")
    else:
        print("   Pas de buffer sauvegardé : redémarrage buffer vide.")
        # même correctif que pour HER : éviter d'échantillonner trop tôt
        model.learning_starts = model.num_timesteps + 1000

    # Rechargement de l'état du curriculum (palier où on s'était arrêté)
    start_stage = 0
    if os.path.exists(STATE_OUT):
        with open(STATE_OUT) as f:
            state = json.load(f)
        start_stage = state.get("stage", 0)
        print(f"   État curriculum rechargé : palier {start_stage+1}/"
              f"{len(CURRICULUM_THRESHOLDS)} ({CURRICULUM_THRESHOLDS[start_stage]*100:.0f} cm).")

    callback = CurriculumCallback(
        env, CURRICULUM_THRESHOLDS, PROMOTION_SUCCESS_RATE,
        MIN_STEPS_PER_STAGE, start_stage=start_stage, verbose=1,
    )
    # on cale le début de palier sur le compteur courant (garde-fou cohérent)
    callback.stage_start_step = model.num_timesteps

    print(f"3. Poursuite de l'apprentissage (+{ADDITIONAL_TIMESTEPS:,} pas)...")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=False,      # CONTINUE le compteur
        progress_bar=True,
    )

    print("4. Sauvegarde (modèle + buffer + état)...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    callback.save_state(STATE_OUT)
    env.close()
    print(f"Terminé. {model.num_timesteps:,} pas au total. "
          f"Palier : {callback.stage+1}/{len(CURRICULUM_THRESHOLDS)} "
          f"({CURRICULUM_THRESHOLDS[callback.stage]*100:.0f} cm).")


if __name__ == "__main__":
    main()
