# ==============================================================================
# FICHIER : fred_phase3_env.py
# RÔLE : PHASE 3 — évitement, départ FIXE (droite) + arrivée ALÉATOIRE (gauche).
#        Obstacle + départ identiques à la phase 2 (hérités). Cible tirée dans
#        la MOITIÉ EXTERNE gauche (jamais près de l'obstacle), au sol, avec
#        REJECTION SAMPLING pour garantir l'atteignabilité (portée UR7e).
#
# Restriction demandée : cibles entre le bord gauche (x_min) et le MILIEU de la
# moitié gauche (x_mid). On évite ainsi les configurations quasi-impossibles
# collées à l'obstacle (peu de poses valides -> collisions -> apprentissage lent).
# ==============================================================================

import numpy as np
from fred_phase2_env import (FredPhase2Env, BOX_DIMS_CM, FLOOR_Z_CM, M_TO_CM)
from fred_phase_base import WORKSPACE

# pré-filtre de portée (rejection sampling) : distance épaule -> cible
SHOULDER_Z_CM = 16.0          # hauteur approx de l'épaule du UR7e
REACH_MAX_CM = 82.0           # seuil prudent (portée nominale ~85, marge de sécu)
MAX_SAMPLE_TRIES = 50


class FredPhase3Env(FredPhase2Env):
    """Phase 3 : départ fixe droite, cible aléatoire moitié EXTERNE gauche, sol."""

    def _reach_from_shoulder_cm(self, x_cm, y_cm, z_cm):
        return float(np.sqrt(x_cm**2 + y_cm**2 + (z_cm - SHOULDER_Z_CM)**2))

    def _target_point_cm(self):
        # bord gauche du volume
        x_outer = WORKSPACE["x"][0] * 100
        # bord de l'obstacle côté gauche
        x_obstacle_edge = -(BOX_DIMS_CM[0] / 2)
        # MILIEU de la moitié gauche : limite interne des cibles (jamais au-delà)
        x_inner = (x_obstacle_edge + x_outer) / 2.0
        # bornes y du volume
        y_lo = WORKSPACE["y"][0] * 100
        y_hi = WORKSPACE["y"][1] * 100

        # rejection sampling : on retire tant que la cible est hors portée
        for _ in range(MAX_SAMPLE_TRIES):
            x_cm = self._rng.uniform(x_outer, x_inner)   # moitié EXTERNE gauche
            y_cm = self._rng.uniform(y_lo, y_hi)
            if self._reach_from_shoulder_cm(x_cm, y_cm, FLOOR_Z_CM) <= REACH_MAX_CM:
                return np.array([x_cm, y_cm, FLOOR_Z_CM])
        # fallback (rare) : on rapproche en y pour garantir l'atteignabilité
        x_cm = self._rng.uniform(x_outer, x_inner)
        return np.array([x_cm, y_lo, FLOOR_Z_CM])
