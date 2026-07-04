"""
Entraînement SAC — Suivi de trajectoire UR7e
──────────────────────────────────────────────
Approche + suivi de lignes aléatoires sur le tableau.

Métriques suivies :
  - taux de complétion (ligne entière suivie)
  - erreur RMS latérale (précision)
  - taux d'accrochage (phase approche réussie)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from ur7e_traj_env import UR7eTrajEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
class TrajMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards   = []
        self.ep_rms       = []
        self.ep_progress  = []
        self.completed    = 0
        self.accrochages  = 0
        self.ep_count     = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_count += 1
                self.ep_rewards.append(info["episode"]["r"])
                # Récupérer les dernières métriques de l'épisode
                self.ep_rms.append(info.get("err_rms_mm", 0))
                self.ep_progress.append(info.get("progression", 0))
                if info.get("completed", False):
                    self.completed += 1
                if info.get("phase", 0) == 1:
                    self.accrochages += 1

                if self.ep_count % 500 == 0:
                    n = min(500, len(self.ep_rewards))
                    print(f"  Ep {self.ep_count:6d} | "
                          f"reward {np.mean(self.ep_rewards[-n:]):8.1f} | "
                          f"RMS {np.mean(self.ep_rms[-n:]):6.1f} mm | "
                          f"prog {np.mean(self.ep_progress[-n:])*100:4.0f}% | "
                          f"complétés {self.completed/self.ep_count*100:4.1f}% | "
                          f"accroch {self.accrochages/self.ep_count*100:4.0f}%")
        return True

# ─────────────────────────────────────────────
N_ENVS = 4
print(f"\nCréation de {N_ENVS} environnements de suivi...")

train_env = make_vec_env(
    lambda: Monitor(UR7eTrajEnv(max_steps=800)),
    n_envs=N_ENVS
)
eval_env = Monitor(UR7eTrajEnv(max_steps=800))

model = SAC(
    "MlpPolicy", train_env, device=device,
    learning_rate=3e-4, buffer_size=1_000_000,
    batch_size=1024, tau=0.005, gamma=0.99,
    ent_coef="auto", train_freq=(1, "step"), gradient_steps=2,
    policy_kwargs=dict(net_arch=[512, 512], activation_fn=torch.nn.ReLU),
    verbose=0, tensorboard_log="./logs_traj/",
)

print(f"SAC : [512,512] | batch 1024 | suivi trajectoire\n")

os.makedirs("models_traj", exist_ok=True)
os.makedirs("checkpoints_traj", exist_ok=True)

callbacks = [
    TrajMetricsCallback(),
    EvalCallback(eval_env, best_model_save_path="./models_traj/",
                 log_path="./logs_traj/", eval_freq=25_000,
                 n_eval_episodes=20, deterministic=True, verbose=1),
    CheckpointCallback(save_freq=300_000, save_path="./checkpoints_traj/",
                       name_prefix="ur7e_traj"),
]

TOTAL = 2_000_000   # suivi plus complexe → plus de steps
print(f"Entraînement — {TOTAL:,} steps\n")

model.learn(total_timesteps=TOTAL, callback=callbacks,
            progress_bar=True, reset_num_timesteps=True)

model.save("models_traj/ur7e_traj_final")
print("\nModèle sauvegardé : models_traj/ur7e_traj_final.zip")

# ─────────────────────────────────────────────
# Évaluation finale
# ─────────────────────────────────────────────
print("\nÉvaluation finale (100 trajectoires)...")
test_env = UR7eTrajEnv(max_steps=800)
rms_list, completed, accroches = [], 0, 0

for _ in range(100):
    obs, _ = test_env.reset()
    done = False
    final_info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = test_env.step(a)
        done = term or trunc
        final_info = info
    rms_list.append(final_info.get("err_rms_mm", 0))
    if final_info.get("completed", False):
        completed += 1
    if final_info.get("phase", 0) == 1:
        accroches += 1

rms_list = np.array(rms_list)
print(f"\n  Taux d'accrochage  : {accroches}%")
print(f"  Taux de complétion : {completed}%")
print(f"  Erreur RMS moyenne : {rms_list.mean():.1f} mm")
print(f"  Erreur RMS médiane : {np.median(rms_list):.1f} mm")
print("\nTerminé !")
