#!/usr/bin/env python3
"""
Visualisation RViz2 — Suivi de trajectoire UR7e
──────────────────────────────────────────────────
Affiche dans RViz2 :
  - le robot UR7e qui suit la ligne (/joint_states)
  - la trajectoire cible complète en VERT (/trajectory_marker)
  - le point cible courant en ROUGE (/target_marker)
  - la trace réelle de l'effecteur en BLEU (/trace_marker)

Reproduit fidèlement la logique de ur7e_traj_env (phases approche + suivi).

Lancement (après robot_state_publisher) :
  ~/venv_ur7e/bin/python viz_traj.py
"""
import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as GeoPoint
from std_msgs.msg import ColorRGBA

UR7E_WS = os.path.expanduser('~/ur7e_ws')
TRAJ_DIR = os.path.join(UR7E_WS, 'trajectoire')
sys.path.insert(0, UR7E_WS)
sys.path.insert(0, TRAJ_DIR)

from stable_baselines3 import SAC
from ur7e_env_v4 import JOINT_LIMITS, HOME_CONFIG, fk, get_chain
from trajectory_generator import generate_trajectory, trajectory_length
from ur7e_traj_env import (
    K_LOOKAHEAD, SEUIL_POINT, SEUIL_ACCROCHE, ERREUR_MAX, JOINT_NAMES
)

MODEL_PATH = os.path.join(TRAJ_DIR, 'models_traj', 'best_model')


class TrajVizNode(Node):
    def __init__(self):
        super().__init__('ur7e_traj_viz')

        self.get_logger().info('Chargement ikpy...')
        get_chain()

        self.get_logger().info(f'Chargement SAC : {MODEL_PATH}')
        self.model = SAC.load(MODEL_PATH)
        self.get_logger().info('Modèle suivi chargé ✓')

        # ── État ────────────────────────────────────
        self.k          = K_LOOKAHEAD
        self.angles     = HOME_CONFIG.copy().astype(np.float32)
        self.trajectory = None
        self.cursor     = 0
        self.phase      = 0          # 0=approche, 1=suivi
        self.max_cursor = 0
        self.stagnation = 0
        self.trace      = []         # historique positions effecteur
        self.errors     = []

        # ── Publishers ──────────────────────────────
        self.pub_js    = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_traj  = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.pub_targ  = self.create_publisher(Marker, '/target_marker', 10)
        self.pub_trace = self.create_publisher(Marker, '/trace_marker', 10)
        self.pub_mur   = self.create_publisher(Marker, '/mur_marker', 10)

        # ── Première trajectoire ────────────────────
        self.new_trajectory()

        # ── Timers ──────────────────────────────────
        self.timer      = self.create_timer(0.02, self.loop)    # 50 Hz contrôle
        self.timer_traj = self.create_timer(0.5, self.publish_trajectory)  # ligne verte
        self.timer_mur  = self.create_timer(1.0, self.publish_mur)            # le mur/tableau

        self.get_logger().info('Node suivi démarré ✓')

    # ──────────────────────────────────────────────
    def new_trajectory(self):
        """Génère une nouvelle ligne et réinitialise l'état."""
        self.trajectory, name = generate_trajectory(n_points=60)
        self.cursor     = 0
        self.phase      = 0
        self.max_cursor = 0
        self.stagnation = 0
        self.trace      = []
        self.errors     = []
        self.angles     = HOME_CONFIG.copy().astype(np.float32)
        length = trajectory_length(self.trajectory) * 100
        self.get_logger().info(
            f'Nouvelle trajectoire : {name} ({length:.0f} cm)'
        )

    # ──────────────────────────────────────────────
    def closest_on_traj(self, pos):
        lo = max(0, self.cursor - 3)
        hi = min(len(self.trajectory), self.cursor + 12)
        window = self.trajectory[lo:hi]
        dists  = np.linalg.norm(window - pos, axis=1)
        idx    = int(np.argmin(dists))
        return lo + idx, float(dists[idx])

    def get_lookahead(self):
        pts = []
        for i in range(self.k):
            idx = min(self.cursor + i, len(self.trajectory) - 1)
            pts.append(self.trajectory[idx])
        return np.concatenate(pts).astype(np.float32)

    def build_obs(self):
        angles_norm = np.array([
            self.angles[i] / JOINT_LIMITS[i][1] for i in range(6)
        ], dtype=np.float32)
        pos_eff     = fk(self.angles)
        progression = np.array([self.cursor / len(self.trajectory)],
                                dtype=np.float32)
        phase       = np.array([float(self.phase)], dtype=np.float32)
        target      = self.trajectory[self.cursor]
        lookahead   = self.get_lookahead()
        obs = np.concatenate([
            angles_norm, pos_eff, progression, phase, target, lookahead
        ]).astype(np.float32)
        return obs, pos_eff

    # ──────────────────────────────────────────────
    def loop(self):
        obs, pos_eff = self.build_obs()

        # Prédiction
        action, _ = self.model.predict(obs, deterministic=True)
        delta = np.clip(action, -1, 1) * 0.05
        new_a = self.angles + delta
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            new_a[i] = np.clip(new_a[i], lo, hi)
        self.angles = new_a.astype(np.float32)

        pos_eff = fk(self.angles)
        self.trace.append(pos_eff.copy())
        if len(self.trace) > 300:
            self.trace.pop(0)

        target     = self.trajectory[self.cursor]
        dist_cible = float(np.linalg.norm(pos_eff - target))
        _, err_lat = self.closest_on_traj(pos_eff)

        # ── Gestion des phases ──
        if self.phase == 0:
            dist_debut = float(np.linalg.norm(pos_eff - self.trajectory[0]))
            if dist_debut < SEUIL_ACCROCHE:
                self.phase = 1
                self.errors = []
                self.get_logger().info('→ Phase SUIVI (ligne accrochée)')
        else:
            self.errors.append(err_lat)
            if dist_cible < SEUIL_POINT and self.cursor < len(self.trajectory)-1:
                self.cursor += 1
                if self.cursor > self.max_cursor:
                    self.max_cursor = self.cursor

            # Fin de trajectoire → nouvelle ligne
            if self.cursor >= len(self.trajectory)-1 and dist_cible < SEUIL_POINT:
                rms = np.sqrt(np.mean(np.square(self.errors))) * 1000
                self.get_logger().info(
                    f'✓ Trajectoire complétée ! RMS = {rms:.1f} mm — '
                    f'nouvelle ligne dans 2s'
                )
                self.publish_state(pos_eff)
                import time; time.sleep(2)
                self.new_trajectory()
                return

            # Sortie de ligne → recommencer
            if err_lat > ERREUR_MAX:
                self.get_logger().warn('✗ Sorti de la ligne — nouvelle tentative')
                import time; time.sleep(1)
                self.new_trajectory()
                return

        self.publish_state(pos_eff)

    # ──────────────────────────────────────────────
    def publish_state(self, pos_eff):
        # joint_states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = JOINT_NAMES
        js.position = self.angles.tolist()
        js.velocity = [0.0]*6
        js.effort   = [0.0]*6
        self.pub_js.publish(js)

        # Point cible courant (sphère rouge)
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'target', 0, Marker.SPHERE
        m.action = Marker.ADD
        t = self.trajectory[self.cursor]
        m.pose.position.x = float(t[0])
        m.pose.position.y = float(t[1])
        m.pose.position.z = float(t[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
        self.pub_targ.publish(m)

        # Trace effecteur (ligne bleue)
        if len(self.trace) > 1:
            tr = Marker()
            tr.header.frame_id = 'base_link'
            tr.header.stamp = self.get_clock().now().to_msg()
            tr.ns, tr.id, tr.type = 'trace', 0, Marker.LINE_STRIP
            tr.action = Marker.ADD
            tr.scale.x = 0.008
            tr.color.r, tr.color.g, tr.color.b, tr.color.a = 0.2, 0.4, 1.0, 0.9
            for p in self.trace:
                pt = GeoPoint()
                pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
                tr.points.append(pt)
            self.pub_trace.publish(tr)

    # ──────────────────────────────────────────────
    def publish_mur(self):
        """Mur (tableau) vertical derrière la trajectoire."""
        from trajectory_generator import (
            TABLEAU_X, TABLEAU_Y_MIN, TABLEAU_Y_MAX,
            TABLEAU_Z_MIN, TABLEAU_Z_MAX
        )
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'mur', 0, Marker.CUBE
        m.action = Marker.ADD

        # Centre du mur — juste derrière le plan de la trajectoire
        marge = 0.15
        cy = (TABLEAU_Y_MIN + TABLEAU_Y_MAX) / 2
        cz = (TABLEAU_Z_MIN + TABLEAU_Z_MAX) / 2
        m.pose.position.x = float(TABLEAU_X + 0.02)   # 2 cm derrière la ligne
        m.pose.position.y = float(cy)
        m.pose.position.z = float(cz)
        m.pose.orientation.w = 1.0

        # Dimensions : fin en X, large en Y et Z
        m.scale.x = 0.02                                       # épaisseur
        m.scale.y = float(TABLEAU_Y_MAX - TABLEAU_Y_MIN + 2*marge)
        m.scale.z = float(TABLEAU_Z_MAX - TABLEAU_Z_MIN + 2*marge)

        # Gris clair semi-transparent
        m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.85, 0.88, 0.6
        self.pub_mur.publish(m)

    def publish_trajectory(self):
        """Ligne verte = trajectoire cible complète."""
        if self.trajectory is None:
            return
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'trajectory', 0, Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.2, 0.8
        for p in self.trajectory:
            pt = GeoPoint()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            m.points.append(pt)
        self.pub_traj.publish(m)

        # Points de la trajectoire (petites sphères)
        mp = Marker()
        mp.header.frame_id = 'base_link'
        mp.header.stamp = self.get_clock().now().to_msg()
        mp.ns, mp.id, mp.type = 'traj_points', 1, Marker.SPHERE_LIST
        mp.action = Marker.ADD
        mp.scale.x = mp.scale.y = mp.scale.z = 0.012
        mp.color.r, mp.color.g, mp.color.b, mp.color.a = 0.0, 0.7, 0.1, 0.5
        for p in self.trajectory:
            pt = GeoPoint()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            mp.points.append(pt)
        self.pub_traj.publish(mp)


def main(args=None):
    rclpy.init(args=args)
    node = TrajVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
