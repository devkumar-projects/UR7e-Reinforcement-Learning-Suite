# ==============================================================================
# FICHIER : bullet_ur7e_hybrid.py
# RÔLE : Moteur de l'architecture HYBRIDE IK -> RL -> IK pour évitement d'un
#        obstacle CYLINDRIQUE (géométrie humaine simplifiée, 1m x Ø50cm).
#
# PIPELINE PAR ÉPISODE :
#   1. cible atteignable 360° + pose de départ
#   2. voie articulaire départ->cible (interpolation des angles + FK)
#   3. cylindre placé pour intercepter le bras sur cette trajectoire
#   4. détection zone de collision -> point d'ENTRÉE (-5cm) et de SORTIE (+5cm)
#   5. téléportation du robot à la config du point d'ENTRÉE (reset RL)
#   6. [RL] contourner pour rejoindre le point de SORTIE (seuil 5cm), en
#      passant de préférence ENTRE l'obstacle et la base (incitation douce)
#   7. (à l'exécution : recalcul IK sortie réelle -> cible, hors entraînement)
#   9. collision du BRAS ENTIER avec le cylindre à tout instant -> fin (échec)
#
# Le moteur expose la phase RL (étapes 5-6). Les tronçons IK (1er et dernier)
# sont déterministes et gérés à l'évaluation, pas pendant l'entraînement RL.
#
# Réutilise les briques éprouvées du moteur 360 : _set_config, _link_points,
# sample_reachable, FK PyBullet. Géométrie cylindre/entrée/sortie/intérieur
# testée isolément avant intégration.
# ==============================================================================

import os
import numpy as np
import pybullet as p
import pybullet_data


class BulletUR7eHybrid:
    """UR7e — phase RL d'évitement d'un cylindre, dans le pipeline IK->RL->IK."""

    CONTROLLABLE_JOINTS = [2, 3, 4, 5, 6, 7]
    EE_LINK_NAME = "tool0"
    SHOULDER_CENTER = np.array([0.0, 0.0, 0.163])
    REACH_MAX = 0.85
    REACH_MIN = 0.18
    BASE_XY = np.array([0.0, 0.0])          # axe vertical de la base du robot

    def __init__(self, gui=False, max_episode_len=300,
                 success_threshold=0.05, urdf_dir=None,
                 ik_tol=0.005, seed=None,
                 cyl_radius=0.25, cyl_height=1.0,
                 traj_samples=40, entry_exit_margin=0.05,
                 collision_penalty=10.0, proximity_margin=0.05,
                 proximity_weight=0.5, interior_weight=0.3,
                 n_link_samples=5):
        """
        Paramètres spécifiques HYBRIDE / cylindre :
          cyl_radius        : rayon du cylindre obstacle (0.25 = Ø50cm).
          cyl_height        : hauteur du cylindre (1.0 m).
          traj_samples      : nb de configs échantillonnées sur la voie articulaire.
          entry_exit_margin : marge (m) avant/après la zone de collision pour
                              définir les points d'entrée et de sortie.
          collision_penalty : pénalité + fin si le bras pénètre le cylindre.
          proximity_margin  : zone tampon (m) de pénalité douce de proximité.
          proximity_weight  : poids de la pénalité de proximité.
          interior_weight   : poids de l'incitation douce à passer côté intérieur.
          n_link_samples    : densité d'échantillonnage du bras.
        """
        self.gui = gui
        self.max_episode_len = max_episode_len
        self.success_threshold = success_threshold
        self.ik_tol = ik_tol
        self.cyl_radius = cyl_radius
        self.cyl_height = cyl_height
        self.traj_samples = traj_samples
        self.entry_exit_margin = entry_exit_margin
        self.collision_penalty = collision_penalty
        self.proximity_margin = proximity_margin
        self.proximity_weight = proximity_weight
        self.interior_weight = interior_weight
        self.n_link_samples = n_link_samples
        self.rng = np.random.default_rng(seed)

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        if urdf_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_dir = os.path.join(script_dir, "ur7e_pybullet")
        self.urdf_dir = urdf_dir

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
            mv = info[11] if info[11] > 0 else 3.14
            self.joint_max_vel.append(mv)
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)
        self.joint_max_vel = np.array(self.joint_max_vel)

        self.home_position = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])

        self.steps = 0
        self.target = None              # point de SORTIE (objectif RL)
        self.final_target = None        # cible finale
        self.cyl_base = None            # centre base cylindre (x,y,0)
        self.entry_config = None        # config articulaire du point d'entrée
        self.exit_point = None          # position cartésienne du point de sortie
        self._target_marker = None
        self._cyl_marker = None

    # ----------------------------------------------------------------------
    #  URDF / liens
    # ----------------------------------------------------------------------
    def _find_link_index(self, name):
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[12].decode() == name:
                return j
        return self.CONTROLLABLE_JOINTS[-1]

    # ----------------------------------------------------------------------
    #  ÉTAT ARTICULAIRE / FK
    # ----------------------------------------------------------------------
    def _get_joint_states(self):
        states = p.getJointStates(self.robot, self.CONTROLLABLE_JOINTS)
        angles = np.array([s[0] for s in states])
        velocities = np.array([s[1] for s in states])
        return angles, velocities

    def get_ee_position(self):
        ls = p.getLinkState(self.robot, self.ee_index,
                            computeForwardKinematics=True)
        return np.array(ls[0])

    def _set_config(self, q):
        for k, j in enumerate(self.CONTROLLABLE_JOINTS):
            p.resetJointState(self.robot, j, q[k], 0.0)

    def _link_points(self):
        """Points échantillonnés le long du bras (liens + interpolations)."""
        pts = []
        link_positions = []
        for j in self.CONTROLLABLE_JOINTS:
            ls = p.getLinkState(self.robot, j, computeForwardKinematics=True)
            link_positions.append(np.array(ls[0]))
        link_positions.append(self.get_ee_position())
        for a, b in zip(link_positions[:-1], link_positions[1:]):
            for t in np.linspace(0.0, 1.0, self.n_link_samples):
                pts.append(a + t * (b - a))
        return np.array(pts)

    # ----------------------------------------------------------------------
    #  GÉOMÉTRIE CYLINDRE (testée isolément)
    # ----------------------------------------------------------------------
    def _point_cylinder_distance(self, P):
        """Distance signée d'un point à la surface du cylindre vertical.
        <0 = à l'intérieur du cylindre."""
        b = self.cyl_base
        dx = P[0] - b[0]; dy = P[1] - b[1]
        radial = np.sqrt(dx * dx + dy * dy)
        d_radial = radial - self.cyl_radius
        d_vert = max(b[2] - P[2], P[2] - (b[2] + self.cyl_height))
        if d_radial <= 0 and d_vert <= 0:
            return max(d_radial, d_vert)
        elif d_radial > 0 and d_vert <= 0:
            return d_radial
        elif d_radial <= 0 and d_vert > 0:
            return d_vert
        else:
            return np.sqrt(d_radial ** 2 + d_vert ** 2)

    def _arm_cylinder_distance(self):
        """Distance min (surface) bras entier <-> cylindre. <0=pénétration."""
        if self.cyl_base is None:
            return None
        pts = self._link_points()
        return min(self._point_cylinder_distance(P) for P in pts)

    def _interior_score(self, ee_xy):
        """>0 si l'effecteur est côté INTÉRIEUR (entre base et obstacle),
        <0 si côté extérieur (au-delà de l'obstacle)."""
        obstacle_xy = self.cyl_base[:2]
        axis = obstacle_xy - self.BASE_XY
        axis_norm = axis / (np.linalg.norm(axis) + 1e-9)
        proj = np.dot(ee_xy - obstacle_xy, axis_norm)
        return -proj

    # ----------------------------------------------------------------------
    #  CIBLE ATTEIGNABLE + IK
    # ----------------------------------------------------------------------
    def sample_reachable(self, max_tries=200):
        """Tire un point cartésien atteignable (validé IK exacte) + sa config."""
        q_current, _ = self._get_joint_states()
        for _ in range(max_tries):
            cand = np.array([
                self.rng.uniform(-self.REACH_MAX, self.REACH_MAX),
                self.rng.uniform(-self.REACH_MAX, self.REACH_MAX),
                self.rng.uniform(0.0, self.SHOULDER_CENTER[2] + self.REACH_MAX),
            ])
            r = np.linalg.norm(cand - self.SHOULDER_CENTER)
            if not (self.REACH_MIN <= r <= self.REACH_MAX):
                continue
            sol = p.calculateInverseKinematics(self.robot, self.ee_index,
                                               cand.tolist())
            self._set_config(np.array(sol))
            ee = self.get_ee_position()
            if np.linalg.norm(ee - cand) < self.ik_tol:
                self._set_config(q_current)
                return cand, np.array(sol)
        self._set_config(q_current)
        return np.array([0.4, 0.0, 0.4]), None

    def _ik(self, point):
        """IK exacte -> config articulaire (6,) pour atteindre `point`."""
        return np.array(p.calculateInverseKinematics(self.robot, self.ee_index,
                                                     list(point)))

    # ----------------------------------------------------------------------
    #  VOIE ARTICULAIRE + DÉTECTION ENTRÉE/SORTIE
    # ----------------------------------------------------------------------
    def _joint_path(self, q_start, q_goal):
        """Voie articulaire = interpolation linéaire des angles (définition B).
        Renvoie un tableau (traj_samples, 6) de configs."""
        ss = np.linspace(0.0, 1.0, self.traj_samples)
        return np.array([(1 - s) * q_start + s * q_goal for s in ss])

    def _cache_arm_points(self, path):
        """Calcule UNE SEULE FOIS les points du bras pour chaque config de la
        voie (cinématique indépendante du cylindre). Renvoie une liste de
        tableaux de points + la position EE de chaque config. C'est le coeur de
        l'optimisation : on ne refait jamais la cinématique pour tester
        différents placements de cylindre."""
        q_save, _ = self._get_joint_states()
        arm_points_per_config = []
        ee_per_config = []
        for q in path:
            self._set_config(q)
            arm_points_per_config.append(self._link_points())
            ee_per_config.append(self.get_ee_position())
        self._set_config(q_save)
        return arm_points_per_config, np.array(ee_per_config)

    def _clearances_from_cache(self, arm_points_per_config):
        """Distances min bras<->cylindre à partir des points du bras MIS EN
        CACHE (aucune cinématique : juste de la géométrie point<->cylindre)."""
        clears = []
        for pts in arm_points_per_config:
            clears.append(min(self._point_cylinder_distance(P) for P in pts))
        return np.array(clears)

    def _ee_at_config(self, q):
        """Position cartésienne de l'effecteur pour une config (FK)."""
        q_save, _ = self._get_joint_states()
        self._set_config(q)
        ee = self.get_ee_position()
        self._set_config(q_save)
        return ee

    def _find_entry_exit_cached(self, path, clearances, ee_path):
        """Comme _find_entry_exit mais utilise ee_path en CACHE (pas de FK).
        Localise la zone de collision et renvoie (entry_idx, exit_idx) avec
        marge, ou None si collision aux extrémités (placement invalide)."""
        mask = clearances < 0.0
        if not mask.any():
            return None
        first = int(np.argmax(mask))
        last = len(mask) - 1 - int(np.argmax(mask[::-1]))
        if first == 0 or last == len(mask) - 1:
            return None
        # longueur cartésienne approx d'une étape (depuis le cache ee_path)
        step_len = (np.linalg.norm(ee_path[-1] - ee_path[0])
                    / max(1, len(path) - 1)) + 1e-9
        margin_steps = max(1, int(self.entry_exit_margin / step_len))
        entry = max(0, first - margin_steps)
        exit_ = min(len(path) - 1, last + margin_steps)
        if clearances[entry] <= 0.0 or clearances[exit_] <= 0.0:
            while entry > 0 and clearances[entry] <= 0.0:
                entry -= 1
            while exit_ < len(path) - 1 and clearances[exit_] <= 0.0:
                exit_ += 1
            if clearances[entry] <= 0.0 or clearances[exit_] <= 0.0:
                return None
        return entry, exit_

    def _place_cylinder_and_cut(self, q_start, q_goal, max_tries=40):
        """Place le cylindre pour intercepter la voie articulaire, puis calcule
        les points d'entrée/sortie. OPTIMISÉ : la cinématique (positions du bras
        pour chaque config) est calculée UNE SEULE FOIS via le cache, puis on
        teste les placements de cylindre par géométrie pure (rapide)."""
        path = self._joint_path(q_start, q_goal)
        # cinématique faite UNE fois (le gros du coût)
        arm_points_per_config, ee_path = self._cache_arm_points(path)
        ee_start = ee_path[0]
        ee_goal = ee_path[-1]

        for _ in range(max_tries):
            frac = self.rng.uniform(0.35, 0.65)
            mid = ee_start + frac * (ee_goal - ee_start)
            jitter = self.rng.uniform(-0.08, 0.08, size=2)
            self.cyl_base = np.array([mid[0] + jitter[0],
                                      mid[1] + jitter[1], 0.0])
            # garde-fou : ne pas engloutir départ ni cible EE
            if (self._point_cylinder_distance(ee_start) < 0.05 or
                    self._point_cylinder_distance(ee_goal) < 0.05):
                continue
            # clearances depuis le CACHE (aucune cinématique refaite)
            clears = self._clearances_from_cache(arm_points_per_config)
            res = self._find_entry_exit_cached(path, clears, ee_path)
            if res is None:
                continue
            entry_idx, exit_idx = res
            return path[entry_idx], ee_path[exit_idx], path[exit_idx]
        return None

    # ----------------------------------------------------------------------
    #  MARQUEURS VISUELS
    # ----------------------------------------------------------------------
    def _spawn_target_marker(self):
        if self._target_marker is not None:
            p.removeBody(self._target_marker)
            self._target_marker = None
        if self.target is None:
            return
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                  rgbaColor=[0.1, 0.9, 0.1, 0.9])
        self._target_marker = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=self.target.tolist())

    def _spawn_cyl_marker(self):
        if self._cyl_marker is not None:
            p.removeBody(self._cyl_marker)
            self._cyl_marker = None
        if self.cyl_base is None:
            return
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.cyl_radius,
                                  length=self.cyl_height,
                                  rgbaColor=[0.9, 0.1, 0.1, 0.40])
        # le centre géométrique du cylindre est à mi-hauteur
        center = [self.cyl_base[0], self.cyl_base[1],
                  self.cyl_base[2] + self.cyl_height / 2.0]
        self._cyl_marker = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis, basePosition=center)

    # ----------------------------------------------------------------------
    #  RESET : prépare un épisode de phase RL (téléporte au point d'entrée)
    # ----------------------------------------------------------------------
    def reset(self, max_setup_tries=30):
        self.steps = 0
        for _ in range(max_setup_tries):
            # 1) pose de départ atteignable + cible finale atteignable
            ee_start, q_start = self.sample_reachable()
            if q_start is None:
                continue
            self.final_target, q_goal = self.sample_reachable()
            if q_goal is None:
                continue
            # voie articulaire départ->cible doit être assez longue
            if np.linalg.norm(self.final_target - ee_start) < 0.30:
                continue
            # 2-4) place le cylindre pour intercepter, calcule entrée/sortie
            self._set_config(q_start)
            res = self._place_cylinder_and_cut(q_start, q_goal)
            if res is None:
                continue
            entry_config, exit_point, exit_config = res
            self.entry_config = entry_config
            self.exit_point = exit_point
            self.target = exit_point          # objectif RL = point de sortie
            # 5) téléporte le robot à la config du point d'entrée
            self._set_config(entry_config)
            # sécurité : le départ RL ne doit pas déjà être en collision
            if self._arm_cylinder_distance() <= 0.0:
                continue
            self._spawn_target_marker()
            self._spawn_cyl_marker()
            for _ in range(5):
                p.stepSimulation()
            return self._get_observation()
        # échec de setup : fallback minimal (rare)
        self._set_config(self.home_position)
        self.target = self.get_ee_position() + np.array([0.1, 0, 0])
        self.cyl_base = np.array([0.5, 0.0, 0.0])
        self.entry_config = self.home_position.copy()
        self.exit_point = self.target.copy()
        self._spawn_target_marker()
        self._spawn_cyl_marker()
        return self._get_observation()

    # ----------------------------------------------------------------------
    #  OBSERVATION (28D) : état + cible(sortie) + obstacle cylindre
    # ----------------------------------------------------------------------
    def _get_observation(self):
        angles, velocities = self._get_joint_states()
        ee = self.get_ee_position()
        to_target = self.target - ee
        # infos cylindre : base (3), rayon+hauteur (2), vecteur ee->axe (2 horiz)
        if self.cyl_base is not None:
            to_axis = np.array([self.cyl_base[0] - ee[0],
                                self.cyl_base[1] - ee[1]])
            cyl_info = np.concatenate([self.cyl_base,
                                       [self.cyl_radius, self.cyl_height],
                                       to_axis])
        else:
            cyl_info = np.zeros(7)
        # 12 + 3(ee) + 3(target) + 3(to_target) + 7(cyl) = 28
        obs = np.concatenate([angles, velocities, ee, self.target,
                              to_target, cyl_info])
        return obs.astype(np.float32)

    # ----------------------------------------------------------------------
    #  STEP : commande vitesse + reward (reaching sortie + évitement + intérieur)
    # ----------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_velocities = action * self.joint_max_vel
        for k, j in enumerate(self.CONTROLLABLE_JOINTS):
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                    targetVelocity=float(target_velocities[k]),
                                    force=150.0)
        p.stepSimulation()
        self.steps += 1

        ee = self.get_ee_position()
        distance = np.linalg.norm(self.target - ee)      # vers le point de SORTIE
        _, velocities = self._get_joint_states()

        # --- shaping de reaching (recette éprouvée) vers le point de sortie ---
        reward = -distance
        reward -= 0.01 * np.linalg.norm(action)
        if distance < 0.03:
            reward += 0.5 * (0.03 - distance) / 0.03
        if distance < 0.05:
            reward -= 0.05 * np.linalg.norm(velocities)

        done = False
        info = {"distance": distance}

        # --- évitement du cylindre (bras entier) ---
        d_obs = self._arm_cylinder_distance()
        if d_obs is not None:
            info["obstacle_clearance"] = d_obs
            if d_obs < 0.0:
                reward -= self.collision_penalty
                done = True
                info["done_reason"] = "collision"
                info["collision"] = 1.0
            elif d_obs < self.proximity_margin:
                reward -= self.proximity_weight * \
                    (self.proximity_margin - d_obs) / self.proximity_margin

        # --- incitation DOUCE au passage côté intérieur (entre base et obstacle) ---
        if not done:
            score = self._interior_score(ee[:2])     # >0 intérieur, <0 extérieur
            # bonus borné, modéré : ne domine pas le reaching
            reward += self.interior_weight * np.tanh(5.0 * score) * 0.1

        # --- succès : point de sortie atteint sans collision ---
        if not done and distance < self.success_threshold:
            reward += 10.0
            done = True
            info["done_reason"] = "exit_reached"
            info["collision"] = info.get("collision", 0.0)

        if self.steps >= self.max_episode_len:
            done = True
            info["done_reason"] = info.get("done_reason", "max_steps")
            info["collision"] = info.get("collision", 0.0)

        return self._get_observation(), reward, done, info

    def close(self):
        p.disconnect(self.cid)
