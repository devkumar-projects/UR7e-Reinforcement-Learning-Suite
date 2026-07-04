# ==============================================================================
# FICHIER : fred_phase2_env.py
# RÔLE : PHASE 2 — évitement d'obstacle, départ ET arrivée FIXES.
#        Boîte fixe (15x27x22 cm) barrant la profondeur, centrée x=0.
#        Départ FIXE : centre moitié droite, au sol. Arrivée FIXE : centre
#        moitié gauche, au sol. Contournement par le HAUT.
# ==============================================================================

import numpy as np
import pybullet as p
from tf_agents.trajectories import time_step as ts
from fred_phase_base import FredPhaseBase, WORKSPACE, FLOOR_Z_CM, M_TO_CM

# Géométrie obstacle (cm) : épaisseur x=15, profondeur y=40 (tout le volume),
# hauteur z=40 ; centré x=0, y=45, base au sol.
# Géométrie obstacle (cm) : épaisseur x=15, profondeur y=27 (toute la profondeur
# du nouveau volume), hauteur z=22 ; centré x=0, y=38.5, base au sol.
# Sommet à z=37cm ; plafond du volume z=58cm -> marge 21cm pour contourner.
BOX_DIMS_CM = (5.0, 27.0, 35.0)
BOX_CENTER_XY_CM = (0.0, 60.5)
# points fixes au sol
RIGHT_X_CM = (BOX_DIMS_CM[0] / 2 + WORKSPACE["x"][1] * 100) / 2
LEFT_X_CM = -RIGHT_X_CM
MID_Y_CM = (WORKSPACE["y"][0] + WORKSPACE["y"][1]) * 100 / 2

# Points fixes phase 2
START_X_CM = RIGHT_X_CM + 10.0
START_Y_CM = MID_Y_CM
START_Z_CM = FLOOR_Z_CM

TARGET_X_CM = LEFT_X_CM - 10.0
TARGET_Y_CM = MID_Y_CM
TARGET_Z_CM = FLOOR_Z_CM


class FredPhase2Env(FredPhaseBase):
    """Phase 2 : A->B fixes, obstacle central barrant la profondeur."""

    def _build_box(self):
        half = [d / 200.0 for d in BOX_DIMS_CM]
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half,
            physicsClientId=self._physics_client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=[0.9, 0.2, 0.2, 0.9],
            physicsClientId=self._physics_client
        )
        center_cm = [
            BOX_CENTER_XY_CM[0],
            BOX_CENTER_XY_CM[1],
            BOX_DIMS_CM[2] / 2.0
        ]
        bid = p.createMultiBody(
            0,
            col,
            vis,
            basePosition=[c / 100.0 for c in center_cm],
            physicsClientId=self._physics_client
        )
        self._obstacle_ids = [bid]

    def _start_cm(self):
        return np.array([START_X_CM, START_Y_CM, START_Z_CM])

    def _target_point_cm(self):
        return np.array([TARGET_X_CM, TARGET_Y_CM, TARGET_Z_CM])

    def _reset(self):
        self._steps = 0
        self._episode_ended = False

        for oid in self._obstacle_ids:
            p.removeBody(oid, physicsClientId=self._physics_client)

        self._obstacle_ids = []
        self._build_box()

        start_m = self._start_cm() / M_TO_CM
        self._set_config(self._ik(start_m))
        self._current_pose_cm = self._get_ee_pos_cm()

        self._target_cm = self._target_point_cm()

        obs, total_distance = self._get_observation()
        self._closest_distance_so_far = total_distance

        return ts.restart(obs)
