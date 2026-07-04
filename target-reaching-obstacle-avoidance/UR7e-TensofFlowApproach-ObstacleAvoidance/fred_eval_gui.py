# ==============================================================================
# FICHIER : fred_eval_gui.py
# RÔLE : Charger une politique entraînée et la visualiser dans PyBullet avec
#        un tracé (trace) en 3D de la trajectoire de l'effecteur.
#
# Usage : python fred_eval_gui.py --phase 2 --policy fred_policy_phase2_final
# ==============================================================================

import argparse
import time
import tensorflow as tf
import pybullet as p

from tf_agents.environments import tf_py_environment

# Import de vos environnements
from fred_phase1_env import FredPhase1Env
from fred_phase2_env import FredPhase2Env
from fred_phase3_env import FredPhase3Env

def main():
    parser = argparse.ArgumentParser(description="Évaluation visuelle et tracé 3D")
    parser.add_argument("--phase", type=int, default=2, help="Phase à charger (1, 2 ou 3)")
    parser.add_argument("--policy", type=str, required=True, 
                        help="Chemin vers le dossier du modèle (ex: fred_policy_phase2_final)")
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes à jouer")
    args = parser.parse_args()

    print(f"\n[*] Chargement de l'environnement Phase {args.phase} en mode GUI...")
    if args.phase == 1:
        py_env = FredPhase1Env(use_gui=True)
    elif args.phase == 2:
        py_env = FredPhase2Env(use_gui=True)
    else:
        py_env = FredPhase3Env(use_gui=True)
        
    # Enveloppement dans l'environnement TensorFlow
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    print(f"[*] Chargement des poids du réseau depuis '{args.policy}'...")
    try:
        policy = tf.saved_model.load(args.policy)
    except Exception as e:
        print(f"[!] Erreur lors du chargement. Vérifiez que le dossier existe bien. Détail: {e}")
        py_env.close()
        return

    print("[*] Début de l'évaluation...")
    
    # Caméra pour bien voir le volume de travail
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=50,
                                 cameraPitch=-30, cameraTargetPosition=[0, 0.38, 0.3],
                                 physicsClientId=py_env._physics_client)

    for ep in range(args.episodes):
        time_step = tf_env.reset()
        
        # Récupération de la position initiale (convertie de cm à mètres pour PyBullet)
        prev_pos = py_env._get_ee_pos_cm() / 100.0
        
        # Effacer les lignes tracées lors de l'épisode précédent
        p.removeAllUserDebugItems(physicsClientId=py_env._physics_client)
        
        # Placer un marqueur textuel sur la cible
        target_pos = py_env._target_cm / 100.0
        p.addUserDebugText("CIBLE", target_pos.tolist(), textColorRGB=[0, 1, 0], 
                           textSize=1.5, physicsClientId=py_env._physics_client)

        step_count = 0
        ep_reward = 0.0

        # Boucle de l'épisode
        while not time_step.is_last():
            # Inférence : Le réseau de neurones choisit la meilleure action (déterministe)
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            
            # Position après avoir appliqué l'action
            curr_pos = py_env._get_ee_pos_cm() / 100.0
            
            # --- TRACER LA TRAJECTOIRE EN 3D ---
            # Ajoute un segment de ligne rouge entre la position précédente et l'actuelle
            p.addUserDebugLine(
                lineFromXYZ=prev_pos.tolist(),
                lineToXYZ=curr_pos.tolist(),
                lineColorRGB=[1, 0.2, 0.2],  # Rouge clair
                lineWidth=2.5,
                lifeTime=0,                  # Reste à l'écran indéfiniment
                physicsClientId=py_env._physics_client
            )
            
            prev_pos = curr_pos
            ep_reward += float(time_step.reward.numpy()[0])
            step_count += 1
            
            # Ralentir la simulation pour que l'œil humain puisse suivre le mouvement
            time.sleep(0.02)
            
        # Bilan de fin d'épisode
        statut = "SUCCÈS" if ep_reward >= 50.0 else "COLLISION" if ep_reward <= -50.0 else "TIMEOUT"
        print(f"Épisode {ep+1}/{args.episodes} terminé : {statut} (Pas: {step_count}, Récompense: {ep_reward:.1f})")
        time.sleep(1.5)  # Petite pause pour admirer la trajectoire finale

    print("[*] Évaluation terminée. Fermeture de PyBullet.")
    py_env.close()

if __name__ == "__main__":
    main()