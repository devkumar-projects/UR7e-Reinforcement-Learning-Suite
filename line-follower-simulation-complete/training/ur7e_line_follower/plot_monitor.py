"""
plot_monitor.py — Tracé temps-réel et post-traitement des courbes de monitoring.

Courbes tracées (inspiré des cours Commande Avancée + MODROB) :
  1. Commandes moteurs q̇_cmd_i (vitesses articulaires envoyées par RL + LQR)
  2. Sorties moteurs q̇_meas_i (vitesses articulaires mesurées = sortie du PI interne)
  3. Erreur statique par axe : e_i(t) = q̇_cmd_i - q̇_meas_i
  4. Position laser FK brute vs EKF filtré (y, z)
  5. Incertitude EKF σ_y, σ_z au cours du temps
  6. Détection KLT caméra (offset_n, klt_confidence, cos_t, sin_t, coverage) depuis /line_detection
  7. Indice de Yoshikawa w(q) (manipulabilité)
  8. Récompense RL par step
  9. RMSE laser↔ligne cumulé

Usage :
  # Monitoring temps-réel (nécessite ROS2 actif)
  ros2 run ur7e_line_follower plot_monitor

  # Post-traitement d'un CSV de métriques enregistré
  python3 -m ur7e_line_follower.plot_monitor --csv path/to/episodes_*.csv
"""
import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # sans affichage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constantes ────────────────────────────────────────────────────────────────
MAX_SAMPLES   = 500    # fenêtre glissante (samples)
PLOT_INTERVAL = 2.0    # secondes entre deux refreshes du graphe
SAVE_DIR      = Path.home() / '.ros' / 'ur7e_line_follower' / 'plots'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

JOINT_NAMES = ['J1_pan', 'J2_shoulder', 'J3_elbow', 'J4_wrist1', 'J5_wrist2', 'J6_wrist3']
COLORS      = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']


# ── Collecteur de données via ROS2 ────────────────────────────────────────────

class DataCollector:
    """
    Souscrit aux topics ROS2 et collecte les données dans des deques circulaires.
    Peut fonctionner sans ROS2 (mode CSV).
    """

    def __init__(self, ros_mode: bool = True):
        self.t_start    = time.time()
        n = MAX_SAMPLES

        # Données temporelles
        self.t           = deque(maxlen=n)

        # Commandes + sorties moteurs (6 axes)
        self.q_dot_cmd   = [deque(maxlen=n) for _ in range(6)]
        self.q_dot_meas  = [deque(maxlen=n) for _ in range(6)]

        # Position laser brute FK vs EKF
        self.laser_y_fk  = deque(maxlen=n)
        self.laser_z_fk  = deque(maxlen=n)
        self.ekf_y       = deque(maxlen=n)
        self.ekf_z       = deque(maxlen=n)
        self.ekf_sig_y   = deque(maxlen=n)
        self.ekf_sig_z   = deque(maxlen=n)

        # Détection KLT caméra — schéma V4 (7 valeurs)
        self.cam_detected   = deque(maxlen=n)
        self.cam_offset_n   = deque(maxlen=n)   # [1] offset_n_norm (was cam_offset_y)
        self.klt_confidence = deque(maxlen=n)   # [2] klt_confidence (was cam_offset_z)
        self.cam_cos_2t     = deque(maxlen=n)   # [3] tangent_cos_theta_directed (NEW)
        self.cam_sin_2t     = deque(maxlen=n)   # [4] tangent_sin_theta_directed (NEW)
        self.cam_coverage   = deque(maxlen=n)   # [5]

        # Manipulabilité
        self.yoshikawa_w    = deque(maxlen=n)

        # Récompense + RMSE
        self.reward         = deque(maxlen=n)
        self.rmse_cumul     = deque(maxlen=n)

        # Dernière commande RL reçue
        self._last_cmd = np.zeros(6)
        self._lock     = threading.Lock()

        if ros_mode:
            self._init_ros()

    def _init_ros(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray, Float32MultiArray
        from sensor_msgs.msg import JointState

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node('plot_monitor_collector')

        # Souscription commandes vitesse RL
        node.create_subscription(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            self._cmd_cb, 10)

        # Souscription états joints (sortie mesurée)
        node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_cb, 10)

        # Détection caméra KLT
        node.create_subscription(
            Float32MultiArray,
            '/line_detection',
            self._cam_cb, 5)

        self._node      = node
        self._executor  = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _cmd_cb(self, msg):
        with self._lock:
            self._last_cmd = np.array(msg.data[:6])

    def _joint_cb(self, msg):
        t_now = time.time() - self.t_start
        with self._lock:
            self.t.append(t_now)
            name_idx = {n: i for i, n in enumerate(msg.name)}

            ordered_vel = [0.0] * 6
            jnames = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                      'wrist_1_joint','wrist_2_joint','wrist_3_joint']
            for k, jn in enumerate(jnames):
                if jn in name_idx:
                    ordered_vel[k] = float(msg.velocity[name_idx[jn]])

            for i in range(6):
                self.q_dot_cmd[i].append(self._last_cmd[i])
                self.q_dot_meas[i].append(ordered_vel[i])

            # Yoshikawa depuis les positions
            try:
                from .kinematics import laser_wall_dot
                from .singularity import yoshikawa
                q_pos = np.array([float(msg.position[name_idx[jn]])
                                   for jn in jnames if jn in name_idx])
                if len(q_pos) == 6:
                    self.yoshikawa_w.append(yoshikawa(q_pos))
                    dot = laser_wall_dot(q_pos)
                    if dot is not None:
                        self.laser_y_fk.append(float(dot[0]))
                        self.laser_z_fk.append(float(dot[1]))
            except Exception:
                pass

    def _cam_cb(self, msg):
        with self._lock:
            if len(msg.data) >= 7:
                self.cam_detected.append(float(msg.data[0]))
                self.cam_offset_n.append(float(msg.data[1]))
                self.klt_confidence.append(float(msg.data[2]))
                self.cam_cos_2t.append(float(msg.data[3]))
                self.cam_sin_2t.append(float(msg.data[4]))
                self.cam_coverage.append(float(msg.data[5]))

    def inject_env_step(self, obs: np.ndarray, reward: float):
        """
        Appelé depuis l'env à chaque step pour enrichir les données.
        EKF se trouve en obs[22:26] dans le schéma V5 (33D).
        """
        with self._lock:
            if len(obs) >= 26:
                self.ekf_y.append(float(obs[22]))
                self.ekf_z.append(float(obs[23]))
                # σ normalisée → σ réelle ≈ σ_norm * 0.05
                self.ekf_sig_y.append(float(obs[24]) * 0.05)
                self.ekf_sig_z.append(float(obs[25]) * 0.05)
            self.reward.append(reward)


# ── Tracé principal ───────────────────────────────────────────────────────────

def make_figure(dc: DataCollector, save_path: Path = None):
    """
    Génère la figure à 9 sous-graphes.
    Retourne la figure matplotlib.
    """
    with dc._lock:
        t      = np.array(dc.t) if dc.t else np.array([0])
        n_t    = len(t)

        # Commandes / mesures moteurs
        cmd  = [np.array(dc.q_dot_cmd[i])  for i in range(6)]
        meas = [np.array(dc.q_dot_meas[i]) for i in range(6)]

        # Laser FK vs EKF
        ly_fk = np.array(dc.laser_y_fk)
        lz_fk = np.array(dc.laser_z_fk)
        ey    = np.array(dc.ekf_y)
        ez    = np.array(dc.ekf_z)
        sy    = np.array(dc.ekf_sig_y)
        sz    = np.array(dc.ekf_sig_z)

        # KLT V4
        cam_det  = np.array(dc.cam_detected)   if dc.cam_detected   else np.array([])
        cam_oy   = np.array(dc.cam_offset_n)   if dc.cam_offset_n   else np.array([])
        cam_conf = np.array(dc.klt_confidence) if dc.klt_confidence else np.array([])
        cam_cov  = np.array(dc.cam_coverage)   if dc.cam_coverage   else np.array([])

        # Yoshikawa
        yw = np.array(dc.yoshikawa_w)
        rew = np.array(dc.reward)

    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(5, 2, hspace=0.45, wspace=0.3)
    fig.suptitle('Monitoring UR7e Line Follower — Commandes, EKF, KLT, Manipulabilité',
                 fontsize=13, fontweight='bold')

    # ── 1. Commandes moteurs q̇_cmd (RL + LQR) ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(6):
        if len(cmd[i]) > 0:
            t_c = t[-len(cmd[i]):]
            ax1.plot(t_c, cmd[i], color=COLORS[i], lw=0.8, label=JOINT_NAMES[i])
    ax1.set_title('Commandes moteurs q̇_cmd [rad/s]\n(sortie RL + correction LQR Riccati)')
    ax1.set_ylabel('rad/s')
    ax1.legend(fontsize=6, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', lw=0.5)

    # ── 2. Sorties moteurs q̇_meas ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(6):
        if len(meas[i]) > 0:
            t_m = t[-len(meas[i]):]
            ax2.plot(t_m, meas[i], color=COLORS[i], lw=0.8, label=JOINT_NAMES[i])
    ax2.set_title('Sorties moteurs q̇_meas [rad/s]\n(vitesses mesurées — sortie PI ros2_control)')
    ax2.set_ylabel('rad/s')
    ax2.legend(fontsize=6, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', lw=0.5)

    # ── 3. Erreur statique par axe e_i = q̇_cmd - q̇_meas ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(6):
        n_e = min(len(cmd[i]), len(meas[i]))
        if n_e > 0:
            err = np.array(cmd[i])[-n_e:] - np.array(meas[i])[-n_e:]
            t_e = t[-n_e:]
            ax3.plot(t_e, err, color=COLORS[i], lw=0.8, label=JOINT_NAMES[i])
    ax3.set_title('Erreur statique par axe : e_i = q̇_cmd − q̇_meas [rad/s]\n'
                  '(doit → 0 avec le correcteur PI interne)')
    ax3.set_ylabel('rad/s')
    ax3.legend(fontsize=6, ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', lw=1.5, ls='--')

    # ── 4. Position laser FK brute vs EKF filtré ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if len(ly_fk) > 0:
        t4 = t[-len(ly_fk):]
        ax4.plot(t4, ly_fk, 'b-', lw=0.8, alpha=0.6, label='y FK brut')
        ax4.plot(t4, lz_fk, 'r-', lw=0.8, alpha=0.6, label='z FK brut')
    if len(ey) > 0:
        t_ey = t[-len(ey):]
        ax4.plot(t_ey, ey, 'b-', lw=1.5, label='y EKF filtré')
        ax4.plot(t_ey, ez, 'r-', lw=1.5, label='z EKF filtré')
    ax4.set_title('Position laser sur le mur — FK brut vs EKF filtré [m]')
    ax4.set_ylabel('Position [m]')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ── 5. Incertitude EKF σ_y, σ_z ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if len(sy) > 0:
        t5 = t[-len(sy):]
        ax5.fill_between(t5, 0, sy * 1000, alpha=0.4, color='blue', label='σ_y [mm]')
        ax5.fill_between(t5, 0, sz * 1000, alpha=0.4, color='red', label='σ_z [mm]')
        ax5.plot(t5, sy * 1000, 'b-', lw=1.0)
        ax5.plot(t5, sz * 1000, 'r-', lw=1.0)
    ax5.set_title('Incertitude EKF σ [mm] — Kalman laser sur mur\n'
                  '(réduit après fusion FK 250Hz + caméra 15Hz)')
    ax5.set_ylabel('σ [mm]')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(5, color='b', lw=0.8, ls=':', alpha=0.7, label='seuil FK 5mm')
    ax5.axhline(15, color='r', lw=0.8, ls=':', alpha=0.7, label='seuil cam 15mm')

    # ── 6. KLT détection caméra V3 ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if len(cam_det) > 0:
        t6 = np.linspace(t[0] if len(t) > 0 else 0, t[-1] if len(t) > 0 else 1, len(cam_det))
        ax6.plot(t6, cam_oy   if len(cam_oy)   == len(t6) else cam_det*0, 'm-', lw=1.0, label='Offset N norm.')
        ax6.plot(t6, cam_conf if len(cam_conf) == len(t6) else cam_det*0, 'c-', lw=1.0, label='KLT conf [0-1]')
        ax6.plot(t6, cam_cov  if len(cam_cov)  == len(t6) else cam_det*0, 'g-', lw=1.0, label='Coverage [0-1]')
        ax6.fill_between(t6, 0, cam_det * 0.1, alpha=0.2, color='orange', label='Détecté')
    ax6.set_title('Détection KLT caméra V3\n(offset_n + klt_confidence + coverage)')
    ax6.set_ylabel('Valeur normalisée')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-1.2, 1.2)

    # ── 7. Indice de Yoshikawa w(q) ────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    if len(yw) > 0:
        t7 = t[-len(yw):]
        ax7.plot(t7, yw, 'purple', lw=1.2, label='w = √det(JJᵀ)')
        ax7.axhline(0.04, color='r', lw=1.0, ls='--', label='W_MIN (pénalité RL)')
        ax7.axhline(0.115, color='g', lw=1.0, ls='--', label='W_HOME (référence)')
        ax7.fill_between(t7, 0, yw, where=np.array(yw) < 0.04,
                         alpha=0.3, color='red', label='Zone singulière')
    ax7.set_title('Indice de Yoshikawa w(q) = √det(JJᵀ)\n'
                  '(§4.11 MODROB — manipulabilité, éviter zones rouges)')
    ax7.set_ylabel('w [-]')
    ax7.set_ylim(0, 0.2)
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # ── 8. Récompense RL par step ──────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    if len(rew) > 0:
        t8 = t[-len(rew):]
        ax8.plot(t8, rew, 'orange', lw=0.8, alpha=0.8)
        # Moyenne glissante
        if len(rew) > 20:
            rew_smooth = np.convolve(rew, np.ones(20)/20, mode='valid')
            ax8.plot(t8[19:], rew_smooth, 'darkorange', lw=2.0, label='Moy. 20 steps')
    ax8.set_title('Récompense RL par step\n(dense : −dist + bonus waypoint + bonus compl. − pénalité sing.)')
    ax8.set_ylabel('Reward')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)
    ax8.axhline(0, color='k', lw=0.5)

    # ── 9. Diagramme espace (y,z) laser FK vs EKF ─────────────────────────
    ax9 = fig.add_subplot(gs[4, :])
    if len(ly_fk) > 0 and len(ey) > 0:
        n9 = min(len(ly_fk), len(ey))
        ax9.plot(ly_fk[-n9:], lz_fk[-n9:], 'b.', ms=1.5, alpha=0.4, label='FK brut')
        ax9.plot(ey[-n9:], ez[-n9:], 'r-', lw=1.0, alpha=0.8, label='EKF filtré')
        # Ellipses d'incertitude à quelques points
        if len(sy) >= 5:
            for k in range(0, min(n9, len(sy)), max(1, n9 // 8)):
                theta = np.linspace(0, 2*np.pi, 30)
                s_y = sy[k] if k < len(sy) else sy[-1]
                s_z = sz[k] if k < len(sz) else sz[-1]
                ax9.plot(ey[k] + s_y * np.cos(theta),
                         ez[k] + s_z * np.sin(theta),
                         'r-', lw=0.5, alpha=0.3)
    ax9.set_title('Trajectoire laser sur le mur — Espace (y, z) [m]\n'
                  '(Bleu = FK brut, Rouge = EKF filtré + ellipses d\'incertitude)')
    ax9.set_xlabel('y [m]')
    ax9.set_ylabel('z [m]')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_aspect('equal', adjustable='box')

    # Ajouter timestamp
    fig.text(0.99, 0.01, time.strftime('%Y-%m-%d %H:%M:%S'),
             ha='right', va='bottom', fontsize=7, color='gray')

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'[plot] Sauvegardé → {save_path}')

    return fig


# ── Mode CSV post-traitement ───────────────────────────────────────────────────

def plot_from_csv(csv_path: str):
    """
    Lit le fichier CSV d'épisodes et trace les courbes d'apprentissage.
    """
    import csv as csv_mod

    timesteps, rewards, dev_means, successes = [], [], [], []
    yoshikawa_vals, rmses, progresses = [], [], []

    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                timesteps.append(int(row['timestep']))
                rewards.append(float(row['reward']))
                dev_means.append(float(row['deviation_mean_m']) * 100)  # cm
                successes.append(float(row['success']) * 100)           # %
                progresses.append(float(row['progress']) * 100)          # %
                if 'yoshikawa_w' in row:
                    yoshikawa_vals.append(float(row['yoshikawa_w']))
                if 'ep_rmse' in row:
                    rmses.append(float(row['ep_rmse']) * 100)           # cm
            except (ValueError, KeyError):
                continue

    if not timesteps:
        print('[plot] Aucune donnée dans le CSV')
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Courbes d\'apprentissage SAC — {Path(csv_path).name}',
                 fontsize=12, fontweight='bold')

    t = np.array(timesteps)

    def smooth(x, w=10):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w)/w, mode='valid'), t[w-1:]

    # Récompense
    ax = axes[0, 0]
    ax.plot(t, rewards, alpha=0.3, color='orange', lw=0.8)
    if len(rewards) >= 10:
        sm, ts = smooth(rewards)
        ax.plot(ts, sm, 'darkorange', lw=2.0)
    ax.set_title('Récompense moyenne par épisode')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    # Écart moyen laser↔ligne
    ax = axes[0, 1]
    ax.plot(t, dev_means, alpha=0.3, color='blue', lw=0.8)
    if len(dev_means) >= 10:
        sm, ts = smooth(dev_means)
        ax.plot(ts, sm, 'darkblue', lw=2.0)
    ax.axhline(4.0, color='r', lw=1.0, ls='--', label='Seuil RMSE 4cm')
    ax.set_title('Écart moyen laser↔ligne [cm]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Écart [cm]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Taux de succès
    ax = axes[1, 0]
    ax.plot(t, successes, alpha=0.3, color='green', lw=0.8)
    if len(successes) >= 10:
        sm, ts = smooth(successes)
        ax.plot(ts, sm, 'darkgreen', lw=2.0)
    ax.set_title('Taux de succès [%]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Succès [%]')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Progression
    ax = axes[1, 1]
    ax.plot(t, progresses, alpha=0.3, color='purple', lw=0.8)
    if len(progresses) >= 10:
        sm, ts = smooth(progresses)
        ax.plot(ts, sm, 'darkviolet', lw=2.0)
    ax.set_title('Progression moyenne sur la ligne [%]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Progression [%]')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Yoshikawa
    ax = axes[2, 0]
    if yoshikawa_vals:
        ax.plot(t[:len(yoshikawa_vals)], yoshikawa_vals, alpha=0.5, color='purple', lw=0.8)
        if len(yoshikawa_vals) >= 10:
            sm, ts = smooth(yoshikawa_vals)
            ax.plot(ts, sm, 'darkviolet', lw=2.0, label='w moyen')
        ax.axhline(0.04, color='r', lw=1.0, ls='--', label='W_MIN')
        ax.axhline(0.115, color='g', lw=1.0, ls='--', label='W_HOME')
        ax.legend(fontsize=7)
    ax.set_title('Indice de Yoshikawa moyen w(q)\n(manipulabilité — éloignement singularités)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('w [-]')
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[2, 1]
    if rmses:
        ax.plot(t[:len(rmses)], rmses, alpha=0.3, color='red', lw=0.8)
        if len(rmses) >= 10:
            sm, ts = smooth(rmses)
            ax.plot(ts, sm, 'darkred', lw=2.0)
        ax.axhline(4.0, color='r', lw=1.5, ls='--', label='Seuil succès 4cm')
        ax.legend(fontsize=7)
    ax.set_title('RMSE laser↔ligne [cm]\n(< 4cm = trajectoire réussie)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('RMSE [cm]')
    ax.grid(True, alpha=0.3)

    out = SAVE_DIR / f'learning_curves_{Path(csv_path).stem}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'[plot] Courbes apprentissage → {out}')
    plt.close(fig)
    return str(out)


# ── Boucle temps-réel ─────────────────────────────────────────────────────────

def run_realtime():
    dc = DataCollector(ros_mode=True)
    print('[plot_monitor] Collecte ROS2 active. Ctrl+C pour arrêter.')
    print(f'[plot_monitor] Graphes sauvegardés dans : {SAVE_DIR}')
    k = 0
    try:
        while True:
            time.sleep(PLOT_INTERVAL)
            k += 1
            save_path = SAVE_DIR / f'monitor_{time.strftime("%H%M%S")}.png'
            make_figure(dc, save_path=save_path)
            print(f'[plot_monitor] Frame {k} → {save_path.name}')
    except KeyboardInterrupt:
        print('[plot_monitor] Arrêt.')


# ── Entry points ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Monitoring UR7e Line Follower')
    parser.add_argument('--csv', type=str, default='',
                        help='Chemin vers un CSV d\'épisodes pour tracé post-entraînement')
    args = parser.parse_args()

    if args.csv:
        plot_from_csv(args.csv)
    else:
        run_realtime()


if __name__ == '__main__':
    main()
