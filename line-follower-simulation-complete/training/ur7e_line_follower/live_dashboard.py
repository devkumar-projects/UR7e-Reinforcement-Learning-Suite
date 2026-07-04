"""
live_dashboard.py — Tableau de bord temps-réel de l'entraînement SAC.

Lit le CSV de métriques en continu et rafraîchit les courbes dans une
fenêtre matplotlib animée.

Usage :
    python3 ~/.ros/ur7e_line_follower/live_dashboard.py
    # ou via ros2 run
    ros2 run ur7e_line_follower live_dashboard
"""
import sys
import time
import glob
import csv
import json
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # GUI interactif
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'

def _resolve_metrics_dir() -> Path:
    latest = DATA_DIR / 'latest_run.txt'
    if latest.exists():
        run_dir = Path(latest.read_text().strip()).expanduser()
        if (run_dir / 'metrics').is_dir():
            return run_dir / 'metrics'
    return DATA_DIR / 'metrics'

METRICS_DIR = _resolve_metrics_dir()

def _requested_total() -> int:
    config = METRICS_DIR.parent / 'config.json'
    if config.exists():
        try:
            return max(1, int(json.loads(config.read_text()).get('total_timesteps_requested', 1)))
        except Exception:
            pass
    return 1

REQUESTED_TOTAL = _requested_total()
REFRESH_MS  = 2000   # rafraîchissement toutes les 2 secondes
WINDOW      = 200    # nombre de points affichés en fenêtre glissante


# ── Lecture CSV ────────────────────────────────────────────────────────────────

def find_latest_csv():
    files = sorted(glob.glob(str(METRICS_DIR / 'train_episodes_*.csv'))) or sorted(glob.glob(str(METRICS_DIR / 'episodes_*.csv')))
    return files[-1] if files else None


def read_csv(path):
    rows = {'timestep': [], 'reward': [], 'dev_mean': [],
            'progress': [], 'success': [], 'yoshikawa': [],
            'ep_rmse': [], 'traj_count': [], 'sing_penalty': []}
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows['timestep'].append(int(row['timestep']))
                    rows['reward'].append(float(row.get('episode_reward', row.get('reward', 'nan'))))
                    rows['dev_mean'].append(float(row['deviation_mean_m']) * 100)
                    rows['progress'].append(float(row['progress']) * 100)
                    rows['success'].append(float(row['success']) * 100)
                    rows['yoshikawa'].append(float(row.get('yoshikawa_wall', row.get('yoshikawa_w', 0))))
                    rows['ep_rmse'].append(float(row.get('ep_rmse_m', row.get('ep_rmse', 0))) * 100)
                    rows['traj_count'].append(int(float(row.get('traj_count', 0))))
                    rows['sing_penalty'].append(float(row.get('sing_penalty', 0)))
                except (ValueError, KeyError):
                    continue
    except (FileNotFoundError, PermissionError):
        pass
    return rows


def smooth(arr, w=10):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid').tolist()


# ── Figure ─────────────────────────────────────────────────────────────────────

def build_dashboard():
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('UR7e Line Follower — Live Training Dashboard')

    gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    style = dict(facecolor='#16213e', framealpha=0.9)
    title_kw = dict(color='white', fontsize=9, fontweight='bold', pad=6)
    tick_kw  = dict(colors='#aaaaaa', labelsize=7)
    grid_kw  = dict(color='#444466', alpha=0.5, lw=0.5)

    axes = []
    for r in range(3):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor('#16213e')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')
            ax.tick_params(axis='both', **tick_kw)
            ax.grid(**grid_kw)
            axes.append(ax)

    fig.suptitle('🤖  SAC Training Dashboard — UR7e Line Follower',
                 color='white', fontsize=13, fontweight='bold', y=0.97)

    return fig, axes, style, title_kw


def setup_axes(axes, title_kw):
    labels = [
        ('Récompense par épisode', 'Reward', None),
        ('Écart laser↔ligne [cm]', 'Écart [cm]', 4.0),
        ('Taux de succès [%]', 'Succès [%]', None),
        ('Progression sur la ligne [%]', 'Progression [%]', None),
        ('Indice de Yoshikawa w(q)', 'w [-]', None),
        ('RMSE laser↔ligne [cm]', 'RMSE [cm]', 4.0),
        ('Trajectoires réussies (count)', 'Traj. réussies', None),
        ('Pénalité singularité', 'Pénalité', None),
        ('Vitesse d\'entraînement [steps/s]', 'Steps/s', None),
    ]
    for ax, (title, ylabel, hline) in zip(axes, labels):
        ax.set_title(title, **title_kw)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=7)
        ax.set_xlabel('Timestep', color='#aaaaaa', fontsize=7)
        if hline is not None:
            ax.axhline(hline, color='#e74c3c', lw=1.0, ls='--', alpha=0.8)


COLORS_MAIN   = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                  '#9b59b6', '#1abc9c', '#e67e22', '#e91e63', '#00bcd4']
COLORS_SMOOTH = ['#ff9999', '#85c1e9', '#82e0aa', '#fad7a0',
                  '#c39bd3', '#76d7c4', '#f0b27a', '#f48fb1', '#80deea']


# ── Animation ──────────────────────────────────────────────────────────────────

_csv_path   = None
_prev_len   = 0
_t0         = time.time()
_prev_steps = 0
_fps_buf    = deque(maxlen=10)


def update(frame, fig, axes):
    global _csv_path, _prev_len, _t0, _prev_steps

    if _csv_path is None:
        _csv_path = find_latest_csv()
        if _csv_path is None:
            axes[0].set_title('En attente du CSV de métriques…',
                              color='yellow', fontsize=9)
            return

    data = read_csv(_csv_path)
    n = len(data['timestep'])

    if n == _prev_len and n > 0:
        return   # pas de nouvelles données

    # Vitesse (steps/s estimée)
    if n > _prev_len and _prev_len > 0:
        dt = time.time() - _t0
        if dt > 0:
            d_steps = data['timestep'][-1] - _prev_steps if data['timestep'] else 0
            _fps_buf.append(d_steps / dt)
    _t0         = time.time()
    _prev_steps = data['timestep'][-1] if data['timestep'] else 0
    _prev_len   = n

    # Fenêtre glissante
    slc = slice(max(0, n - WINDOW), n)

    series = [
        data['reward'],
        data['dev_mean'],
        data['success'],
        data['progress'],
        data['yoshikawa'],
        data['ep_rmse'],
        data['traj_count'],
        data['sing_penalty'],
        list(_fps_buf),
    ]

    t_all  = np.array(data['timestep'])
    t_slc  = t_all[slc]

    for i, (ax, raw) in enumerate(zip(axes, series)):
        ax.clear()
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        ax.grid(color='#444466', alpha=0.5, lw=0.5)

        if not raw:
            continue

        # Pour le FPS (derniers échantillons, pas indexés sur timestep)
        if i == 8:
            fps_arr = list(_fps_buf)
            if fps_arr:
                ax.plot(range(len(fps_arr)), fps_arr,
                        color=COLORS_MAIN[i], lw=1.0, alpha=0.7)
                ax.set_title(f'Vitesse  {fps_arr[-1]:.1f} steps/s (moy {np.mean(fps_arr):.1f})',
                             color='white', fontsize=9, fontweight='bold', pad=6)
            ax.set_ylabel('Steps/s', color='#aaaaaa', fontsize=7)
            ax.set_xlabel('Historique', color='#aaaaaa', fontsize=7)
            continue

        raw_slc = raw[slc] if len(raw) >= n else raw
        sm  = smooth(raw_slc, w=min(10, max(1, len(raw_slc) // 5)))
        t_sm = t_slc[len(t_slc) - len(sm):]

        ax.plot(t_slc, raw_slc, color=COLORS_MAIN[i], lw=0.7, alpha=0.4)
        if len(sm) > 1:
            ax.plot(t_sm, sm, color=COLORS_SMOOTH[i], lw=1.8)

        # Ligne seuil
        thresholds = {1: 4.0, 5: 4.0}
        if i in thresholds:
            ax.axhline(thresholds[i], color='#e74c3c', lw=1.0, ls='--', alpha=0.8)

        # Yoshikawa : zones
        if i == 4:
            ax.axhline(0.04, color='#e74c3c', lw=1.0, ls=':', alpha=0.9, label='W_MIN')
            ax.axhline(0.115, color='#2ecc71', lw=0.8, ls=':', alpha=0.7, label='W_HOME')

        # Titre dynamique avec valeur courante
        val_now = raw[-1] if raw else 0
        titles_fmt = [
            f'Récompense  {val_now:+.2f}',
            f'Écart  {val_now:.1f} cm',
            f'Succès  {val_now:.0f}%',
            f'Progression  {val_now:.0f}%',
            f'Yoshikawa w  {val_now:.3f}',
            f'RMSE  {val_now:.1f} cm',
            f'Traj. réussies  {int(val_now)}',
            f'Pénalité sing.  {val_now:.3f}',
            '',
        ]
        ax.set_title(titles_fmt[i], color='white', fontsize=9, fontweight='bold', pad=6)
        ax.set_ylabel(['Reward','cm','%','%','w','cm','n','val',''][i],
                      color='#aaaaaa', fontsize=7)
        ax.set_xlabel('Timestep', color='#aaaaaa', fontsize=7)

    # Titre global avec progression
    total = data['timestep'][-1] if data['timestep'] else 0
    requested = max(REQUESTED_TOTAL, total, 1)
    pct = min(total / requested, 1.0) * 100
    fig.suptitle(
        f'🤖  SAC Training Dashboard — {total:,} / {requested:,} steps  ({pct:.1f}%)   '
        f'[épisodes: {n}]',
        color='white', fontsize=12, fontweight='bold', y=0.97)


def main():
    fig, axes, style, title_kw = build_dashboard()
    setup_axes(axes, title_kw)

    csv_path = find_latest_csv()
    if csv_path:
        print(f'[dashboard] CSV : {csv_path}')
    else:
        print(f'[dashboard] En attente de données dans {METRICS_DIR}')

    ani = FuncAnimation(fig, update, fargs=(fig, axes),
                        interval=REFRESH_MS, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    main()
