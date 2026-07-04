# ==============================================================================
# FICHIER : fred_phase_base.py
# RÔLE : Environnement de BASE commun aux 3 phases (transfer learning UR7e/Fred).
#        Observation 15D CONSTANTE sur les 3 phases (répulsifs nuls si pas
#        d'obstacle) -> permet de reprendre directement les poids d'une phase
#        à l'autre.
#
# Commande en POSE effecteur (Δpose), IK PyBullet derrière (fidèle à Fred).
# Observation 15D : attractif effecteur(3) + répulsifs 3 points(9) + pose norm(3).
# 3 points de contrôle : effecteur (tool0), poignet (wrist_1), coude (forearm).
# Collision = BRAS ENTIER via getClosestPoints (couvre tous les segments).
#
# Volume de travail devant le robot (m) : x[-0.42,0.42] y[0.25,0.52] z[0.15,0.58].
# ==============================================================================

import os
import numpy as np
import pybullet as p
import pybullet_data

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from fred_potential_fields import (
    get_control_point_pos, get_attractive_force_world,
    get_repulsive_forces_world, normalize_vector_for_obs, M_TO_CM,
)

#WORKSPACE = {"x": (-0.42, 0.42), "y": (0.25, 0.52), "z": (0.15, 0.58)}
WORKSPACE = {"x": (-0.42, 0.42), "y": (0.45, 0.82), "z": (0.15, 0.58)}
FLOOR_Z_CM = WORKSPACE["z"][0] * 100.0       # "au sol" = plancher du volume (15 cm)


class FredPhaseBase(py_environment.PyEnvironment):
    """Base commune : reaching par pose + champs de potentiel, observation 15D."""

    def __init__(self, use_gui=False, urdf_dir=None, max_steps=150,
                 target_reached_distance_cm=5.0, collision_margin_cm=1.0,
                 ik_max_iter=20, ik_residual=1e-4, seed=None):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._use_gui = use_gui
        self._max_steps = max_steps
        self._target_reached_distance = target_reached_distance_cm
        self._collision_margin_cm = collision_margin_cm
        # paramètres IK (coût/précision du solveur) — défaut = comportement std
        self._ik_max_iter = ik_max_iter
        self._ik_residual = ik_residual

        self._xyz_step_cm = 3.0
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype=np.float32, minimum=-5.0, maximum=5.0,
            name="observation")

        self._physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self._physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        if urdf_dir is None:
            urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "ur7e_pybullet")
        cwd = os.getcwd()
        os.chdir(urdf_dir)
        self._robot = p.loadURDF("ur7e.urdf", [0, 0, 0],
                                 p.getQuaternionFromEuler([0, 0, 0]),
                                 useFixedBase=True,
                                 physicsClientId=self._physics_client)
        os.chdir(cwd)

        self._controllable_joints = [2, 3, 4, 5, 6, 7]
        self._ee_index = self._find_link("tool0")
        # 3 points de contrôle (fidèle à Fred) : tool0, wrist_1_link, forearm_link
        self._control_links = [self._find_link("tool0"),
                               self._find_link("wrist_1_link"),
                               self._find_link("forearm_link")]
        self._link_radii = np.array([5.0, 5.0, 6.0])
        self._link_weights = np.array([1.0, 1.0, 1.0])

        # pose de repos standard (config home)
        self._home = np.array([0.0, -1.0, 1.0, -1.57, -1.57, 0.0])

        self._steps = 0
        self._target_cm = None
        self._current_pose_cm = None
        self._closest_distance_so_far = None
        self._episode_ended = False
        self._obstacle_ids = []
        self._last_terminal_reward = 0.0
        self._last_distance_cm = 0.0

    # ------------------------------------------------------------------ helpers
    def _find_link(self, name):
        for i in range(p.getNumJoints(self._robot,
                                      physicsClientId=self._physics_client)):
            info = p.getJointInfo(self._robot, i,
                                  physicsClientId=self._physics_client)
            if info[12].decode() == name:
                return i
        return self._controllable_joints[-1]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _set_config(self, q):
        for k, j in enumerate(self._controllable_joints):
            p.resetJointState(self._robot, j, q[k], 0.0,
                              physicsClientId=self._physics_client)

    def _get_ee_pos_cm(self):
        ls = p.getLinkState(self._robot, self._ee_index,
                            physicsClientId=self._physics_client)
        return np.array(ls[0]) * M_TO_CM

    def _ik(self, point_m):
        # IK paramétrable : maxNumIterations / residualThreshold contrôlent le
        # coût (et la précision) du solveur. Réduire maxNumIterations accélère
        # chaque pas mais rend la conversion pose->angles moins précise.
        # Valeurs par défaut (_ik_max_iter=20, _ik_residual=1e-4) = comportement
        # PyBullet standard (équivalent au code d'origine).
        return np.array(p.calculateInverseKinematics(
            self._robot, self._ee_index, list(point_m),
            maxNumIterations=self._ik_max_iter,
            residualThreshold=self._ik_residual,
            physicsClientId=self._physics_client))

    def _sample_workspace_point_m(self):
        return np.array([self._rng.uniform(*WORKSPACE["x"]),
                         self._rng.uniform(*WORKSPACE["y"]),
                         self._rng.uniform(*WORKSPACE["z"])])

    # ------------------------------------------------------------------ obstacle
    def _in_collision(self):
        """Collision du BRAS ENTIER avec un obstacle (getClosestPoints)."""
        if not self._obstacle_ids:
            return False
        margin_m = self._collision_margin_cm / 100.0
        for oid in self._obstacle_ids:
            pts = p.getClosestPoints(bodyA=self._robot, bodyB=oid,
                                     distance=margin_m,
                                     physicsClientId=self._physics_client)
            for cp in pts:
                if cp[8] < 0.0:
                    return True
        return False

    # ------------------------------------------------------------------ obs 15D
    def _get_observation(self):
        cp = np.array([get_control_point_pos(self._robot, li,
                                             self._physics_client)
                       for li in self._control_links])
        ee_pos = cp[0]
        attractive_cutoff = 10.0
        attr_forces, total_distance = get_attractive_force_world(
            np.array([ee_pos]), np.array([self._target_cm]),
            attractive_cutoff_distance=attractive_cutoff)
        attr_ee = attr_forces[0] / attractive_cutoff

        if self._obstacle_ids:
            rep_forces = get_repulsive_forces_world(
                self._robot, self._control_links, self._link_radii,
                self._link_weights, self._obstacle_ids, self._physics_client)
        else:
            rep_forces = np.zeros((3, 3))

        pose_norm = (self._current_pose_cm / 40.0).tolist()
        obs = []
        obs += normalize_vector_for_obs(attr_ee)
        obs += normalize_vector_for_obs(rep_forces[0])
        obs += normalize_vector_for_obs(rep_forces[1])
        obs += normalize_vector_for_obs(rep_forces[2])
        obs += pose_norm
        return np.array(obs, dtype=np.float32), total_distance

    # ------------------------------------------------------------------ step
    def _apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_pose_cm = self._current_pose_cm + self._xyz_step_cm * action
        target_pose_cm[0] = np.clip(target_pose_cm[0],
                                    WORKSPACE["x"][0]*M_TO_CM, WORKSPACE["x"][1]*M_TO_CM)
        target_pose_cm[1] = np.clip(target_pose_cm[1],
                                    WORKSPACE["y"][0]*M_TO_CM, WORKSPACE["y"][1]*M_TO_CM)
        target_pose_cm[2] = np.clip(target_pose_cm[2],
                                    WORKSPACE["z"][0]*M_TO_CM, WORKSPACE["z"][1]*M_TO_CM)
        q = self._ik(target_pose_cm / M_TO_CM)
        self._set_config(q)
        self._current_pose_cm = self._get_ee_pos_cm()

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        self._apply_action(action)
        self._steps += 1
        obs, total_distance = self._get_observation()
        self._last_distance_cm = total_distance

        progress = self._closest_distance_so_far - total_distance
        if total_distance < self._closest_distance_so_far:
            self._closest_distance_so_far = total_distance
        reward = float(progress)

        if self._in_collision():
            self._episode_ended = True
            self._last_terminal_reward = -50.0
            return ts.termination(obs, reward=-50.0)
        if total_distance < self._target_reached_distance:
            self._episode_ended = True
            speed_bonus = max(0.0, (self._max_steps - self._steps)) * 0.1
            self._last_terminal_reward = 50.0 + speed_bonus
            return ts.termination(obs, reward=float(50.0 + speed_bonus))
        if self._steps >= self._max_steps:
            self._episode_ended = True
            self._last_terminal_reward = reward
            return ts.termination(obs, reward=reward)
        return ts.transition(obs, reward=reward, discount=1.0)

    def close(self):
        p.disconnect(self._physics_client)
