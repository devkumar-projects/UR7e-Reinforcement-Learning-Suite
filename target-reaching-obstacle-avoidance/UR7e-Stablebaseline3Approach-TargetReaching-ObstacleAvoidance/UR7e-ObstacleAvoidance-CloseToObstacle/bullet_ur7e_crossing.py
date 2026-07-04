# ==============================================================================
# FICHIER : bullet_ur7e_crossing.py
# RÔLE : Moteur RL pour le FRANCHISSEMENT d'un obstacle cylindrique.
#
# OBJECTIF : l'agent part d'une zone proche de l'obstacle, côté "départ", et
# doit franchir l'obstacle pour rejoindre la "zone d'évitement réussie"
# (l'autre demi-plan), sans collision et en gardant une marge >= 5 cm.
#
# Géométrie par épisode :
#   - cylindre vertical : H=0.50 m, rayon 0.10 m, position XY aléatoire atteignable.
#   - droite XY d'angle theta aléatoire passant par le centre -> 2 demi-plans.
#     Côté base du robot = "départ/collision" ; côté opposé = "évitement réussi".
#   - départ : effecteur sur la moitié extérieure d'un demi-cylindre virtuel
#     (rayon 0.15 m) côté départ, à >= 5 cm de la surface de l'obstacle.
#   - config initiale via IK, validée non singulière (manipulabilité Yoshikawa).
#
# Action : vitesses articulaires (cohérent avec tout le projet).
# Succès : effecteur dans la zone d'évitement, marge >= 5 cm, sans collision,
#          dans les limites articulaires.
# Échec  : collision / marge critique / hors limites / singularité / max pas.
#
# Briques géométriques (demi-plan, tirage départ, progression) testées isolément.
# ==============================================================================

import os
import numpy as np
import pybullet as p
import pybullet_data


class BulletUR7eCrossing:
    """UR7e — franchissement d'un obstacle cylindrique (zone départ -> évitement)."""

    CONTROLLABLE_JOINTS = [2, 3, 4, 5, 6, 7]
    EE_LINK_NAME = "tool0"
    SHOULDER_CENTER = np.array([0.0, 0.0, 0.163])
    REACH_MAX = 0.85
    REACH_MIN = 0.18
    BASE_XY = np.array([0.0, 0.0])

    def __init__(self, gui=False, max_episode_len=300, urdf_dir=None,
                 ik_tol=0.005, seed=None,
                 cyl_radius=0.10, cyl_height=0.50,
                 virt_radius=0.15, safety_margin=0.05,
                 manip_threshold=0.02,
                 collision_penalty=10.0, success_bonus=10.0,
                 proximity_weight=1.0, progress_weight=2.0,
                 interior_weight=0.3, wrong_side_penalty=2.0,
                 action_penalty=0.01, step_penalty=0.002,
                 n_link_samples=6,
                 field_d0=0.25, field_eta=0.02, field_fmax=2.0):
        """
        cyl_radius/cyl_height : obstacle (0.10 m / 0.50 m).
        virt_radius           : rayon du demi-cylindre virtuel de départ (0.15 m).
        safety_margin         : marge de sécurité (5 cm) sous laquelle on pénalise,
                                et seuil de marge finale pour valider le succès.
        manip_threshold       : seuil de manipulabilité (singularité).
        collision_penalty     : pénalité (+fin) si collision bras<->cylindre.
        success_bonus         : bonus (+fin) si zone d'évitement atteinte avec marge.
        proximity_weight      : poids de la pénalité de proximité (< marge).
        progress_weight       : poids de la récompense de progression vers la zone.
        action_penalty        : poids de la pénalité sur la norme d'action (fluidité).
        step_penalty          : petite pénalité par pas (trajectoires courtes).
        n_link_samples        : densité d'échantillonnage du bras.
        """
        self.gui = gui
        self.max_episode_len = max_episode_len
        self.ik_tol = ik_tol
        self.cyl_radius = cyl_radius
        self.cyl_height = cyl_height
        self.virt_radius = virt_radius
        self.safety_margin = safety_margin
        self.manip_threshold = manip_threshold
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.proximity_weight = proximity_weight
        self.progress_weight = progress_weight
        self.interior_weight = interior_weight
        self.wrong_side_penalty = wrong_side_penalty
        self.action_penalty = action_penalty
        self.step_penalty = step_penalty
        self.n_link_samples = n_link_samples
        self.field_d0 = field_d0
        self.field_eta = field_eta
        self.field_fmax = field_fmax
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

        # état épisode
        self.steps = 0
        self.cyl_base = None            # centre base cylindre (x,y,0)
        self.zone_normal = None         # normale orientée : zone évitement = côté>0
        self.theta = None
        self._prev_side = None          # pour la récompense de progression
        self._cyl_marker = None

    # ----------------------------------------------------------------------
    def _find_link_index(self, name):
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[12].decode() == name:
                return j
        return self.CONTROLLABLE_JOINTS[-1]

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

    def _manipulability(self, q):
        self._set_config(q)
        zeros = [0.0] * len(self.CONTROLLABLE_JOINTS)
        jac_t, jac_r = p.calculateJacobian(
            self.robot, self.ee_index, [0, 0, 0],
            list(q), zeros, zeros)
        J = np.vstack([np.array(jac_t), np.array(jac_r)])
        return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

    # ----------------------------------------------------------------------
    #  GÉOMÉTRIE CYLINDRE + DEMI-PLAN (briques testées isolément)
    # ----------------------------------------------------------------------
    def _point_cylinder_distance(self, P):
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
        if self.cyl_base is None:
            return None
        pts = self._link_points()
        return min(self._point_cylinder_distance(P) for P in pts)

    def _side(self, P_xy):
        """Signe du demi-plan : >0 = zone d'évitement, <0 = zone départ."""
        return float(np.dot(P_xy - self.cyl_base[:2], self.zone_normal))

    def _setup_zones(self, theta):
        """Définit la frontière L (angle theta), oriente la normale n vers la
        zone d'évitement (côté opposé à la base), et calcule la tangente t et
        base_lateral pour la 'moitié intérieure' (droite M perpendiculaire à L
        passant par le centre du cylindre). Renvoie False si la géométrie est
        ambiguë (base quasi alignée avec M -> re-tirage conseillé)."""
        n = np.array([-np.sin(theta), np.cos(theta)])
        if np.dot(self.BASE_XY - self.cyl_base[:2], n) > 0:
            n = -n
        self.zone_normal = n
        self.zone_tangent = np.array([-n[1], n[0]])      # t, direction de L
        self.base_lateral = float(np.dot(self.BASE_XY - self.cyl_base[:2],
                                         self.zone_tangent))
        self.theta = theta
        # ambiguïté : base quasi sur la droite M -> 'intérieur' indéfini
        return abs(self.base_lateral) >= 0.03

    def _inner_side(self, P_xy):
        """>0 si P est dans la MOITIÉ INTÉRIEURE de la zone d'évitement (même
        côté latéral que la base), <0 si moitié extérieure."""
        return float(np.sign(self.base_lateral)
                     * np.dot(P_xy - self.cyl_base[:2], self.zone_tangent))

    # ----------------------------------------------------------------------
    #  TIRAGE OBSTACLE + DÉPART
    # ----------------------------------------------------------------------
    def _sample_obstacle(self):
        """Position XY du cylindre dans l'espace atteignable (anneau autour
        de la base, à portée du bras)."""
        for _ in range(200):
            ang = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(0.30, 0.65)        # distance base->obstacle
            xy = np.array([r * np.cos(ang), r * np.sin(ang)])
            return np.array([xy[0], xy[1], 0.0])
        return np.array([0.5, 0.0, 0.0])

    def _sample_start_on_virtual(self, max_tries=200):
        """Tire une position de départ de l'effecteur sur la moitié extérieure
        du demi-cylindre virtuel (rayon virt_radius) côté DÉPART, à >= marge de
        la surface de l'obstacle. Renvoie (P, q0) avec config IK valide non
        singulière, ou (None, None)."""
        q_save, _ = self._get_joint_states()
        for _ in range(max_tries):
            phi = self.rng.uniform(0, 2 * np.pi)
            dir_xy = np.array([np.cos(phi), np.sin(phi)])
            cand_xy = self.cyl_base[:2] + self.virt_radius * dir_xy
            # doit être côté DÉPART (signe < 0)
            if self._side(cand_xy) >= 0:
                continue
            # marge à la surface de l'obstacle
            radial = np.linalg.norm(cand_xy - self.cyl_base[:2])
            if radial - self.cyl_radius < self.safety_margin:
                continue
            z = self.rng.uniform(0.10, self.cyl_height - 0.05)
            P = np.array([cand_xy[0], cand_xy[1], z])
            # atteignable ?
            rr = np.linalg.norm(P - self.SHOULDER_CENTER)
            if not (self.REACH_MIN <= rr <= self.REACH_MAX):
                continue
            sol = p.calculateInverseKinematics(self.robot, self.ee_index,
                                               P.tolist())
            q0 = self._wrap_to_limits(np.array(sol))
            # rejeter si, même après normalisation, des angles sortent des bornes
            if np.any(q0 < self.joint_lower) or np.any(q0 > self.joint_upper):
                continue
            self._set_config(q0)
            ee = self.get_ee_position()
            if np.linalg.norm(ee - P) > self.ik_tol:
                continue
            # non singulière ?
            if self._manipulability(q0) < self.manip_threshold:
                continue
            # le bras initial ne doit pas déjà être en collision
            self._set_config(q0)
            if self._arm_cylinder_distance() < self.safety_margin:
                continue
            return P, q0
        self._set_config(q_save)
        return None, None

    # ----------------------------------------------------------------------
    def _wrap_to_limits(self, q):
        """Ramène chaque angle dans [-pi, pi], puis tente de le faire rentrer
        dans ses bornes articulaires en ajoutant/retirant 2*pi (récupère les
        solutions IK 'enroulées' qui pointent au même endroit mais sortent des
        bornes)."""
        q = np.array(q, dtype=float)
        out = q.copy()
        for k in range(len(out)):
            a = (out[k] + np.pi) % (2 * np.pi) - np.pi      # -> [-pi, pi]
            lo, hi = self.joint_lower[k], self.joint_upper[k]
            # essaie a, a+2pi, a-2pi pour rester dans [lo, hi]
            for cand in (a, a + 2 * np.pi, a - 2 * np.pi):
                if lo <= cand <= hi:
                    a = cand
                    break
            out[k] = a
        return out

    def _spawn_cyl_marker(self):
        if self._cyl_marker is not None:
            p.removeBody(self._cyl_marker)
            self._cyl_marker = None
        if self.cyl_base is None:
            return
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.cyl_radius,
                                  length=self.cyl_height,
                                  rgbaColor=[0.9, 0.1, 0.1, 0.45])
        center = [self.cyl_base[0], self.cyl_base[1],
                  self.cyl_base[2] + self.cyl_height / 2.0]
        self._cyl_marker = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis, basePosition=center)

    # ----------------------------------------------------------------------
    #  RESET
    # ----------------------------------------------------------------------
    def reset(self, max_setup_tries=40):
        self.steps = 0
        for _ in range(max_setup_tries):
            self.cyl_base = self._sample_obstacle()
            # re-tire theta tant que la géométrie 'intérieur' est ambiguë
            if not self._setup_zones(self.rng.uniform(0, 2 * np.pi)):
                continue
            P, q0 = self._sample_start_on_virtual()
            if q0 is None:
                continue
            self._set_config(q0)
            ee = self.get_ee_position()
            self._prev_side = self._side(ee[:2])
            self._spawn_cyl_marker()
            for _ in range(5):
                p.stepSimulation()
            return self._get_observation()
        # fallback minimal
        self.cyl_base = np.array([0.5, 0.0, 0.0])
        self._setup_zones(0.0)
        self._set_config(np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0]))
        self._prev_side = self._side(self.get_ee_position()[:2])
        self._spawn_cyl_marker()
        return self._get_observation()

    # ----------------------------------------------------------------------
    #  OBSERVATION (~24D)
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    #  CHAMP DE POTENTIEL (répulsifs links + attractifs zone/couloir)
    # ----------------------------------------------------------------------
    def _link_positions(self):
        """Positions cartésiennes des 6 points de contrôle du bras
        (les 6 links contrôlés, le dernier étant l'effecteur tool0)."""
        pos = []
        for j in self.CONTROLLABLE_JOINTS:
            ls = p.getLinkState(self.robot, j, computeForwardKinematics=True)
            pos.append(np.array(ls[0]))
        return pos                       # 6 positions

    def _cylinder_surface_normal(self, P):
        """Pour un point P, renvoie (clearance, normale unitaire obstacle->P).
        La normale combine la composante radiale et verticale selon la zone."""
        b = self.cyl_base
        dx = P[0] - b[0]; dy = P[1] - b[1]
        radial = np.sqrt(dx * dx + dy * dy)
        clearance = self._point_cylinder_distance(P)
        # direction radiale horizontale (de l'axe vers P)
        if radial > 1e-9:
            radial_dir = np.array([dx / radial, dy / radial, 0.0])
        else:
            radial_dir = np.array([1.0, 0.0, 0.0])
        # composante verticale si on est au-dessus/dessous des faces
        z_rel = P[2] - b[2]
        if z_rel < 0:
            vert_dir = np.array([0.0, 0.0, -1.0])
        elif z_rel > self.cyl_height:
            vert_dir = np.array([0.0, 0.0, 1.0])
        else:
            vert_dir = np.zeros(3)
        normal = radial_dir + vert_dir
        n = np.linalg.norm(normal)
        normal = normal / n if n > 1e-9 else radial_dir
        return clearance, normal

    def _repulsive_vector(self, P):
        """Champ répulsif borné type Khatib pour un point P.
        force = eta*(1/d - 1/d0) si d<d0, plafonnée à fmax ; vecteur=force*normale."""
        clearance, normal = self._cylinder_surface_normal(P)
        d = max(clearance, 1e-3)
        if d >= self.field_d0:
            return np.zeros(3)
        force = self.field_eta * (1.0 / d - 1.0 / self.field_d0)
        force = min(force, self.field_fmax)
        return force * normal

    def _attractive_zone(self, ee_xy):
        """Vecteur attractif (3D, z=0) vers le demi-plan d'évitement."""
        side = np.dot(ee_xy - self.cyl_base[:2], self.zone_normal)
        pull = max(0.0, -side)
        v2 = np.tanh(pull) * self.zone_normal
        return np.array([v2[0], v2[1], 0.0])

    def _attractive_inner(self, ee_xy):
        """Vecteur attractif (3D, z=0) vers la moitié intérieure (couloir)."""
        s = np.sign(self.base_lateral)
        inner = s * np.dot(ee_xy - self.cyl_base[:2], self.zone_tangent)
        pull = max(0.0, -inner)
        direction = s * self.zone_tangent
        v2 = np.tanh(pull) * direction
        return np.array([v2[0], v2[1], 0.0])

    def _get_observation(self):
        angles, velocities = self._get_joint_states()
        ee = self.get_ee_position()
        d_arm = self._arm_cylinder_distance()
        d_arm = 0.0 if d_arm is None else d_arm
        to_axis = np.array([self.cyl_base[0] - ee[0],
                            self.cyl_base[1] - ee[1]])
        side_sign = self._side(ee[:2])
        inner_sign = self._inner_side(ee[:2])
        # --- champ de potentiel ---
        # 6 vecteurs répulsifs (un par link) = 18D
        repulsive = np.concatenate([self._repulsive_vector(P)
                                    for P in self._link_positions()])
        # 2 vecteurs attractifs (zone + couloir intérieur) = 6D
        attr_zone = self._attractive_zone(ee[:2])
        attr_inner = self._attractive_inner(ee[:2])
        obs = np.concatenate([
            angles, velocities, ee,                 # 6 + 6 + 3
            self.cyl_base,                          # 3
            [self.cyl_radius, self.cyl_height],     # 2
            to_axis,                                # 2
            [d_arm],                                # 1
            [side_sign],                            # 1
            [inner_sign],                           # 1
            repulsive,                              # 18 (6 links x 3)
            attr_zone,                              # 3
            attr_inner,                             # 3
        ])                                          # total 49
        return obs.astype(np.float32)

    # ----------------------------------------------------------------------
    #  STEP + RÉCOMPENSE MULTI-TERMES
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
        angles, velocities = self._get_joint_states()
        cur_side = self._side(ee[:2])
        d_arm = self._arm_cylinder_distance()

        reward = 0.0
        done = False
        info = {}

        # (a) progression vers la zone d'évitement (et NON vers un point :
        #     pas de biais rectiligne, l'agent peut courber sa trajectoire)
        reward += self.progress_weight * (cur_side - self._prev_side)
        self._prev_side = cur_side

        # (a') moitié intérieure : incitation douce + pénalité FORTE du mauvais
        #      côté (SANS terminer l'épisode, pour ne pas décourager le
        #      franchissement de la frontière).
        inner = self._inner_side(ee[:2])           # >0 intérieur, <0 extérieur
        in_evit = cur_side > 0.0                    # déjà côté évitement ?
        # incitation douce continue vers l'intérieur (bornée, modérée)
        reward += self.interior_weight * np.tanh(5.0 * inner) * 0.1
        # pénalité forte si dans la zone d'évitement mais MAUVAISE moitié
        if in_evit and inner < 0.0:
            reward -= self.wrong_side_penalty

        # (b) pénalité d'action (fluidité) + petite pénalité par pas
        reward -= self.action_penalty * np.linalg.norm(action)
        reward -= self.step_penalty

        # (c) proximité / collision avec le cylindre
        info["obstacle_clearance"] = d_arm
        if d_arm < 0.0:
            reward -= self.collision_penalty
            done = True
            info["done_reason"] = "collision"
            info["collision"] = 1.0
        elif d_arm < self.safety_margin:
            # pénalité progressive quand on entame la marge de sécurité
            reward -= self.proximity_weight * \
                (self.safety_margin - d_arm) / self.safety_margin

        # (d) limites articulaires
        if not done and (np.any(angles < self.joint_lower) or
                         np.any(angles > self.joint_upper)):
            reward -= self.collision_penalty
            done = True
            info["done_reason"] = "joint_limits"
            info["collision"] = info.get("collision", 0.0)

        # (e) singularité dangereuse
        if not done:
            w = self._manipulability(angles)
            if w < self.manip_threshold:
                reward -= self.collision_penalty
                done = True
                info["done_reason"] = "singularity"
                info["collision"] = info.get("collision", 0.0)

        # (f) SUCCÈS : zone d'évitement atteinte DANS LA MOITIÉ INTÉRIEURE
        #     (la plus proche de la base radialement) ET marge de sécurité.
        if not done and cur_side > 0.0 and inner >= 0.0 \
                and d_arm >= self.safety_margin:
            reward += self.success_bonus
            done = True
            info["done_reason"] = "crossed"
            info["collision"] = info.get("collision", 0.0)

        if self.steps >= self.max_episode_len and not done:
            done = True
            info["done_reason"] = "max_steps"
            info["collision"] = info.get("collision", 0.0)

        # distance "au franchissement" : combien il reste pour passer côté>0
        # (négatif = encore côté départ). Utile pour le moniteur.
        info["distance"] = max(0.0, -cur_side)
        return self._get_observation(), reward, done, info

    def close(self):
        p.disconnect(self.cid)
