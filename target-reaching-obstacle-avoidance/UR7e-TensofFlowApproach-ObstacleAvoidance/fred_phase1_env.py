# ==============================================================================
# FICHIER : fred_phase1_env.py
# RÔLE : PHASE 1 — atteinte de cible simple, SANS obstacle.
#        Départ : pose de repos standard. Cible : aléatoire dans le volume.
# ==============================================================================

from tf_agents.trajectories import time_step as ts
from fred_phase_base import FredPhaseBase, M_TO_CM


class FredPhase1Env(FredPhaseBase):
    """Phase 1 : reaching depuis la pose de repos vers une cible aléatoire."""

    def _reset(self):
        self._steps = 0
        self._episode_ended = False
        self._obstacle_ids = []                      # pas d'obstacle
        # départ : TOUJOURS la pose de repos standard
        self._set_config(self._home)
        self._current_pose_cm = self._get_ee_pos_cm()
        # cible : aléatoire dans le volume
        self._target_cm = self._sample_workspace_point_m() * M_TO_CM
        obs, total_distance = self._get_observation()
        self._closest_distance_so_far = total_distance
        return ts.restart(obs)
