# ==============================================================================
# FICHIER : bullet_ur7e.py
# RÔLE : Moteur physique de la simulation du cobot UR7e (Le "Corps" du projet).
#
# DESCRIPTION :
# Simule le bras collaboratif Universal Robots UR7e (6 axes) sous PyBullet.
# Adapté depuis la version UR30. Le UR7e réutilise la géométrie du UR5e
# (mêmes longueurs de segments) ; portée maximale ~850 mm, charge utile 7.5 kg.
#
# Responsabilités :
#   1. Charger le modèle UR7e (URDF + meshes ur5e), base boulonnée au sol.
#   2. Piloter les 6 articulations rotatives en CONTRÔLE DE VITESSE.
#   3. Renvoyer l'état : angles + vitesses des joints, pose de l'effecteur tool0.
#   4. Gérer une cible 3D (tâche reaching) et le calcul de récompense.
#
# CHOIX TECHNIQUES (à justifier à l'oral) :
#   - useFixedBase=True : bras industriel boulonné.
#   - Contrôle en VITESSE : compromis entre couple (trop dur) et position
#     (trop proche d'une commande classique).
#   - Effecteur = link "tool0".
#   - Espace de travail des cibles calibré sur la portée réelle (~850 mm).
# ==============================================================================

import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class BulletUR7e:
    """Moteur physique brut du UR7e. Aucune intelligence ici."""

    # Indices à confirmer via check_ur7e_joints.py
    CONTROLLABLE_JOINTS = [2, 3, 4, 5, 6, 7]  # 6 axes REVOLUTE
    EE_LINK_NAME = "tool0"                     # effecteur = point de travail réel

    def __init__(self, gui=False, max_episode_len=300,
                 success_threshold=0.005, urdf_dir=None):
        self.gui = gui
        self.max_episode_len = max_episode_len
        self.success_threshold = success_threshold

        if urdf_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_dir = os.path.join(script_dir, "ur7e_pybullet")
        self.urdf_dir = urdf_dir

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.dt = 1.0 / 240.0
        p.setTimeStep(self.dt)

        p.loadURDF("plane.urdf")

        cwd = os.getcwd()
        os.chdir(self.urdf_dir)
        self.robot = p.loadURDF("ur7e.urdf", [0, 0, 0],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        os.chdir(cwd)

        self.ee_index = self._find_link_index(self.EE_LINK_NAME)

        self.joint_lower, self.joint_upper, self.joint_max_vel = [], [], []
        for j in self.CONTROLLABLE_JOINTS:
            info = p.getJointInfo(self.robot, j)
            self.joint_lower.append(info[8])
            self.joint_upper.append(info[9])
            mv = info[11] if info[11] > 0 else 3.0
            self.joint_max_vel.append(mv)
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)
        self.joint_max_vel = np.array(self.joint_max_vel)

        # Pose de repos (coude plié, configuration de travail)
        self.home_position = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])

        self.target = None
        self.steps = 0

    def _find_link_index(self, link_name):
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            if info[12].decode("utf-8") == link_name:
                return i
        print(f"[WARN] link '{link_name}' introuvable, fallback dernier joint.")
        return self.CONTROLLABLE_JOINTS[-1]

    def _get_joint_states(self):
        states = p.getJointStates(self.robot, self.CONTROLLABLE_JOINTS)
        angles = np.array([s[0] for s in states])
        velocities = np.array([s[1] for s in states])
        return angles, velocities

    def get_ee_position(self):
        link_state = p.getLinkState(self.robot, self.ee_index,
                                    computeForwardKinematics=True)
        return np.array(link_state[0])

    def sample_target(self):
        """
        Cible 3D dans l'enveloppe atteignable du UR7e (portée ~850 mm).
        Boîte calibrée pour rester dans l'espace de travail réel.
        """
        x = np.random.uniform(0.25, 0.65)
        y = np.random.uniform(-0.45, 0.45)
        z = np.random.uniform(0.15, 0.70)
        return np.array([x, y, z])

    def _spawn_target_marker(self):
        if not self.gui:
            return
        if hasattr(self, "_target_marker") and self._target_marker is not None:
            p.removeBody(self._target_marker)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                  rgbaColor=[1, 0, 0, 0.8])
        self._target_marker = p.createMultiBody(baseMass=0,
                                                baseVisualShapeIndex=vis,
                                                basePosition=self.target)

    def reset(self):
        self.steps = 0
        for k, j in enumerate(self.CONTROLLABLE_JOINTS):
            p.resetJointState(self.robot, j, self.home_position[k], 0.0)
        self.target = self.sample_target()
        self._spawn_target_marker()
        for _ in range(10):
            p.stepSimulation()
        return self._get_observation()

    def _get_observation(self):
        angles, velocities = self._get_joint_states()
        ee = self.get_ee_position()
        to_target = self.target - ee
        obs = np.concatenate([angles, velocities, ee, self.target, to_target])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_velocities = action * self.joint_max_vel

        for k, j in enumerate(self.CONTROLLABLE_JOINTS):
            p.setJointMotorControl2(
                self.robot, j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[k],
                force=500,
            )

        for _ in range(4):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.dt)

        self.steps += 1

        ee = self.get_ee_position()
        distance = np.linalg.norm(self.target - ee)

        # --- Reward simple et robuste (version validée sur UR30) ----------
        # Terme dense : se rapprocher de la cible. Petite pénalité d'action
        # pour la fluidité. Gros bonus terminal à l'atteinte.
        # NB : on n'ajoute PAS de pénalité de vitesse ni de bonus de proximité
        # car ils créaient un effet pervers (l'agent apprenait à NE PAS
        # s'approcher pour éviter les malus), saboteant l'apprentissage.
        reward = -distance
        reward -= 0.01 * np.linalg.norm(action)

        done = False
        info = {"distance": distance}

        if distance < self.success_threshold:
            reward += 10.0
            done = True
            info["done_reason"] = "target_reached"

        if self.steps >= self.max_episode_len:
            done = True
            info["done_reason"] = info.get("done_reason", "max_steps")

        return self._get_observation(), reward, done, info

    def close(self):
        if p.isConnected():
            p.disconnect()
