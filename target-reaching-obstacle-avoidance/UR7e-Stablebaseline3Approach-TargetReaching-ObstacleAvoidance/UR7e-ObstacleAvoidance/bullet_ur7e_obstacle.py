# ==============================================================================
# FICHIER : bullet_ur7e_360.py
# RÔLE : Moteur physique de RÉFÉRENCE ("étalon") pour l'étude sur l'espace de
#        travail COMPLET du UR7e (360° autour de la base).
#
# Différences avec bullet_ur7e.py (l'étalon de l'expérience 100k, inchangé) :
#   1. Cibles échantillonnées dans TOUT l'espace atteignable (360° en x,y),
#      validées par CINÉMATIQUE INVERSE exacte (IK) — pas seulement l'avant.
#   2. Pose de départ ALÉATOIRE à chaque reset (randomisation de l'état initial),
#      avec REJET des configurations singulières (manipulabilité de Yoshikawa)
#      et des AUTO-COLLISIONS. Cela force l'agent à apprendre une politique
#      générale, valable depuis n'importe quelle configuration et vers 360°.
#   3. Au TEST (cartographie), on impose une pose de départ FIXE et unique pour
#      la reproductibilité : passer random_start=False.
#
# Tout le reste (contrôle en vitesse, observation 21D, récompense) est identique
# à l'étalon original, pour que la comparaison reste cohérente.
# ==============================================================================

import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class BulletUR7e360:
    """Moteur physique du UR7e — version espace complet (étalon 360°)."""

    CONTROLLABLE_JOINTS = [2, 3, 4, 5, 6, 7]   # 6 axes REVOLUTE (confirmés)
    EE_LINK_NAME = "tool0"                      # effecteur = point de travail réel

    # Centre approximatif de l'épaule (origine de la sphère d'atteinte),
    # surélevé par rapport à la base au sol.
    SHOULDER_CENTER = np.array([0.0, 0.0, 0.163])
    REACH_MAX = 0.85      # allonge maximale du UR7e (m)
    REACH_MIN = 0.18      # rayon mort central (m) : le bras ne peut se replier davantage

    def __init__(self, gui=False, max_episode_len=300,
                 success_threshold=0.05, urdf_dir=None,
                 random_start=False, ik_tol=0.005,
                 manip_threshold=0.02, seed=None,
                 obstacle_radius=0.10, obstacle_seg_frac=(0.35, 0.65),
                 collision_penalty=10.0, proximity_margin=0.05,
                 proximity_weight=0.5, n_link_samples=8):
        """
        Paramètres OBSTACLE (ajout par rapport au moteur shaped) :
          obstacle_radius   : rayon de la sphère obstacle (m). 0.10 = significatif.
          obstacle_seg_frac : fraction (min,max) du segment effecteur_initial->cible
                              où placer le centre de l'obstacle (tirée aléatoirement).
          collision_penalty : pénalité (et fin d'épisode) si le bras pénètre la sphère.
          proximity_margin  : distance (m) sous laquelle une pénalité douce de
                              proximité s'applique (au-delà du rayon).
          proximity_weight  : poids de la pénalité douce de proximité.
          n_link_samples    : nb de points échantillonnés le long du bras pour
                              estimer la distance bras<->obstacle.
        """
        self.gui = gui
        self.max_episode_len = max_episode_len
        self.success_threshold = success_threshold
        self.random_start = random_start
        self.ik_tol = ik_tol
        self.manip_threshold = manip_threshold
        self.rng = np.random.RandomState(seed)

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

        # Limites effectives pour le tirage des poses de départ : on évite les
        # 10 % extrêmes de chaque articulation (configurations pathologiques).
        # NB : les joints UR ont des limites à ±2π ; on borne à ±π pour rester
        # dans des poses physiquement raisonnables.
        safe = np.minimum(np.abs(self.joint_lower), np.pi)
        safe = np.minimum(safe, np.abs(self.joint_upper))
        self.start_low = -0.9 * safe
        self.start_high = 0.9 * safe

        # Pose home FIXE (utilisée au test, random_start=False)
        self.home_position = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])

        self.target = None
        self.steps = 0

        # --- état OBSTACLE ---
        self.obstacle_radius = obstacle_radius
        self.obstacle_seg_frac = obstacle_seg_frac
        self.collision_penalty = collision_penalty
        self.proximity_margin = proximity_margin
        self.proximity_weight = proximity_weight
        self.n_link_samples = n_link_samples
        self.obstacle_pos = None            # centre de la sphère (3,)
        self.obstacle_marker_id = None      # id visuel pybullet
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

    # ----------------------------------------------------------------------
    def _set_config(self, q):
        """Place instantanément les 6 joints à la configuration q (rad)."""
        for k, j in enumerate(self.CONTROLLABLE_JOINTS):
            p.resetJointState(self.robot, j, q[k], 0.0)

    def _manipulability(self, q):
        """
        Manipulabilité de Yoshikawa w = sqrt(det(J·Jᵀ)) à la configuration q.
        w -> 0 près d'une singularité. On assemble la jacobienne 6x6 via PyBullet.
        """
        self._set_config(q)
        zeros = [0.0] * len(self.CONTROLLABLE_JOINTS)
        # calculateJacobian exige les positions de TOUS les DOF mobiles.
        # Ici nos 6 joints contrôlables sont les seuls mobiles.
        jac_t, jac_r = p.calculateJacobian(
            self.robot, self.ee_index,
            localPosition=[0, 0, 0],
            objPositions=list(q),
            objVelocities=zeros,
            objAccelerations=zeros,
        )
        J = np.vstack([np.array(jac_t), np.array(jac_r)])   # 6x6
        w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        return w

    def _self_collision(self):
        """True si le bras est en auto-collision dans sa configuration courante."""
        p.performCollisionDetection()
        contacts = p.getContactPoints(self.robot, self.robot)
        # on ignore les contacts entre liens adjacents (toujours présents)
        for c in contacts:
            link_a, link_b = c[3], c[4]
            if abs(link_a - link_b) > 1:
                return True
        return False

    def _sample_start_config(self, max_tries=200):
        """
        Tire une pose de départ aléatoire SÛRE : ni singulière (manipulabilité
        suffisante), ni en auto-collision. Restaure une pose neutre si échec.
        """
        for _ in range(max_tries):
            q = self.rng.uniform(self.start_low, self.start_high)
            w = self._manipulability(q)
            if w < self.manip_threshold:
                continue                      # trop proche d'une singularité
            self._set_config(q)
            if self._self_collision():
                continue                      # auto-collision
            return q
        # garde-fou : si aucune pose valide trouvée, on retombe sur home
        print("[WARN] pose de départ aléatoire non trouvée, fallback home.")
        return self.home_position.copy()

    # ----------------------------------------------------------------------
    def sample_target(self, max_tries=500):
        """
        Échantillonne une cible 3D ATTEIGNABLE dans TOUT l'espace (360°),
        validée par cinématique inverse exacte.

        Méthode : tirer une candidate dans la boîte symétrique, pré-filtrer par
        la coquille sphérique (rapide), puis valider par IK (exacte). On restaure
        la pose courante du bras après le test IK pour ne pas la perturber.
        """
        # mémorise la config courante pour la restaurer après les tests IK
        q_current, _ = self._get_joint_states()

        for _ in range(max_tries):
            cand = np.array([
                self.rng.uniform(-self.REACH_MAX, self.REACH_MAX),
                self.rng.uniform(-self.REACH_MAX, self.REACH_MAX),
                self.rng.uniform(0.0, self.SHOULDER_CENTER[2] + self.REACH_MAX),
            ])
            # pré-filtre géométrique : dans la coquille atteignable
            r = np.linalg.norm(cand - self.SHOULDER_CENTER)
            if not (self.REACH_MIN <= r <= self.REACH_MAX):
                continue
            # validation IK exacte
            sol = p.calculateInverseKinematics(self.robot, self.ee_index,
                                               cand.tolist())
            self._set_config(np.array(sol))
            ee = self.get_ee_position()
            if np.linalg.norm(ee - cand) < self.ik_tol:
                self._set_config(q_current)   # restaure la config d'origine
                return cand
        # garde-fou
        self._set_config(q_current)
        print("[WARN] cible atteignable non trouvée, fallback devant le robot.")
        return np.array([0.4, 0.0, 0.4])

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

    # ----------------------------------------------------------------------
    #  OBSTACLE
    # ----------------------------------------------------------------------
    def _link_points(self):
        """
        Échantillonne des points le long du bras (positions des liens contrôlés
        + effecteur) pour estimer la distance bras<->obstacle. On interpole aussi
        entre liens consécutifs pour ne pas rater une collision au milieu d'un
        segment de bras.
        """
        pts = []
        # positions des liens articulés
        link_positions = []
        for j in self.CONTROLLABLE_JOINTS:
            ls = p.getLinkState(self.robot, j, computeForwardKinematics=True)
            link_positions.append(np.array(ls[0]))
        link_positions.append(self.get_ee_position())
        # points interpolés entre liens consécutifs
        for a, b in zip(link_positions[:-1], link_positions[1:]):
            for t in np.linspace(0.0, 1.0, self.n_link_samples):
                pts.append(a + t * (b - a))
        return np.array(pts)

    def _arm_obstacle_distance(self):
        """Distance minimale (surface) entre le bras et la sphère obstacle.
        Négative si pénétration. None si pas d'obstacle."""
        if self.obstacle_pos is None:
            return None
        pts = self._link_points()
        d_centers = np.linalg.norm(pts - self.obstacle_pos, axis=1).min()
        return d_centers - self.obstacle_radius     # <0 => pénétration

    def _place_obstacle(self, ee_start):
        """
        Place l'obstacle sur le segment effecteur_initial -> cible, à une
        fraction aléatoire de l'intervalle obstacle_seg_frac. Garantit que ni le
        départ ni la cible ne sont engloutis (sinon tâche impossible) en
        re-tirant la fraction si besoin.
        """
        for _ in range(20):
            frac = self.rng.uniform(*self.obstacle_seg_frac)
            center = ee_start + frac * (self.target - ee_start)
            # l'obstacle ne doit engloutir ni le départ ni la cible
            if (np.linalg.norm(center - ee_start) > self.obstacle_radius + 0.03 and
                    np.linalg.norm(center - self.target) > self.obstacle_radius + 0.03):
                # vérifie aussi que le BRAS INITIAL entier est dégagé (sinon
                # l'épisode démarrerait en collision, ingérable)
                self.obstacle_pos = center
                if self._arm_obstacle_distance() > 0.02:
                    return
        # garde-fou : milieu du segment
        self.obstacle_pos = ee_start + 0.5 * (self.target - ee_start)

    def _spawn_obstacle_marker(self):
        """(Ré)affiche la sphère obstacle en rouge (GUI only) + collision shape."""
        if self.obstacle_marker_id is not None:
            p.removeBody(self.obstacle_marker_id)
            self.obstacle_marker_id = None
        if self.obstacle_pos is None:
            return
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.obstacle_radius,
                                  rgbaColor=[0.9, 0.1, 0.1, 0.45])
        # corps purement visuel (pas de collision physique : on gère la collision
        # géométriquement via _arm_obstacle_distance, plus robuste et contrôlable)
        self.obstacle_marker_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=self.obstacle_pos.tolist())

    # ----------------------------------------------------------------------
    def reset(self):
        self.steps = 0
        # 1) pose de départ : aléatoire (entraînement) ou fixe (test)
        if self.random_start:
            q0 = self._sample_start_config()
        else:
            q0 = self.home_position.copy()
        self._set_config(q0)
        # 2) cible atteignable (validée IK) — tirée APRÈS avoir posé le bras,
        #    et sample_target restaure la config de départ ensuite.
        self.target = self.sample_target()
        self._set_config(q0)                  # garantit le départ exact
        self._spawn_target_marker()
        # 3) obstacle sur le segment effecteur_initial -> cible
        ee_start = self.get_ee_position()
        self._place_obstacle(ee_start)
        self._spawn_obstacle_marker()
        for _ in range(10):
            p.stepSimulation()
        return self._get_observation()

    def _get_observation(self):
        angles, velocities = self._get_joint_states()
        ee = self.get_ee_position()
        to_target = self.target - ee
        # --- infos OBSTACLE ---
        if self.obstacle_pos is not None:
            to_obstacle = self.obstacle_pos - ee
            obs_pos = self.obstacle_pos
            obs_r = np.array([self.obstacle_radius])
        else:
            to_obstacle = np.zeros(3)
            obs_pos = np.zeros(3)
            obs_r = np.array([0.0])
        # 12 (angles+vit) + 3 (ee) + 3 (cible) + 3 (vers cible)
        #  + 3 (pos obstacle) + 1 (rayon) + 3 (vers obstacle) = 28
        obs = np.concatenate([angles, velocities, ee, self.target, to_target,
                              obs_pos, obs_r, to_obstacle])
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
        _, velocities = self._get_joint_states()

        # --- Reward shaping COMPLET (recette du modèle 100%) ---
        # (1) gradient dense : se rapprocher de la cible
        reward = -distance
        # (2) pénalité d'amplitude d'action : mouvements sobres
        reward -= 0.01 * np.linalg.norm(action)
        # (3) bonus de proximité fine sous 3 cm : densifie le signal là où
        #     l'approche terminale se joue
        if distance < 0.03:
            reward += 0.5 * (0.03 - distance) / 0.03
        # (4) pénalité de vitesse sous 5 cm : force la DÉCÉLÉRATION en approche
        #     finale (c'est ce terme qui produit le "freinage" propre et permet
        #     de conclure au lieu d'osciller/dépasser)
        if distance < 0.05:
            reward -= 0.05 * np.linalg.norm(velocities)

        done = False
        info = {"distance": distance}

        # --- ÉVITEMENT D'OBSTACLE ---
        d_obs = self._arm_obstacle_distance()    # distance surface bras<->sphère
        if d_obs is not None:
            info["obstacle_clearance"] = d_obs
            if d_obs < 0.0:
                # COLLISION : le bras a pénétré la sphère -> grosse pénalité + fin
                reward -= self.collision_penalty
                done = True
                info["done_reason"] = "collision"
                info["collision"] = 1.0
            elif d_obs < self.proximity_margin:
                # zone tampon : pénalité douce, croît quand on s'approche
                reward -= self.proximity_weight * (self.proximity_margin - d_obs) \
                    / self.proximity_margin

        if not done and distance < self.success_threshold:
            reward += 10.0                       # (5) bonus terminal
            done = True
            info["done_reason"] = "target_reached"
            info["collision"] = info.get("collision", 0.0)

        if self.steps >= self.max_episode_len:
            done = True
            info["done_reason"] = info.get("done_reason", "max_steps")

        return self._get_observation(), reward, done, info

    def close(self):
        if p.isConnected():
            p.disconnect()
