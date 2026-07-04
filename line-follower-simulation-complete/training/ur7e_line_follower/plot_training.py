"""
plot_training.py — Générateur de courbes complet post-entraînement pour soutenance.

Produit ~25 figures PNG dans ~/soutenance/figures/ couvrant :
  RL     : reward, succès, progression, RMSE, durée épisode, curriculum
  SAC    : actor/critic loss, entropy, n_updates
  Reward : décomposition des 11 composantes
  Kalman : σ EKF, NIS, evolution temporelle
  KLT    : confiance, détection, laser visible
  LQR    : normes de commande (raw, lqr, null, out)
  MGD    : Yoshikawa, conditionnement
  Traj   : 30 snapshots 2D (cible bleue vs laser rouge), RMSE évolution
  3D     : trajectoire 3D step-y-z (si plotly disponible)
  Distrib: histogrammes RMSE, succès par curriculum, scatter reward vs RMSE

Usage :
    python3 -m ur7e_line_follower.plot_training
    python3 -m ur7e_line_follower.plot_training --out ~/soutenance/figures
    python3 -m ur7e_line_follower.plot_training --metrics ~/.ros/ur7e_line_follower/runs/<run>/metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# ── Chemins par défaut ─────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / '.ros' / 'ur7e_line_follower'
LEGACY_METRICS_DIR = DATA_DIR / 'metrics'
OUT_DIR     = Path.home() / 'soutenance' / 'figures'


def _resolve_metrics_dir(value: str | None = None) -> Path:
    """Resolve an explicit run/metrics directory or the latest isolated run."""
    if value:
        candidate = Path(value).expanduser()
        if (candidate / 'metrics').is_dir():
            return candidate / 'metrics'
        return candidate
    latest_file = DATA_DIR / 'latest_run.txt'
    if latest_file.exists():
        run_dir = Path(latest_file.read_text().strip()).expanduser()
        if (run_dir / 'metrics').is_dir():
            return run_dir / 'metrics'
    return LEGACY_METRICS_DIR

STYLE = {
    'reward':     '#e67e22',
    'success':    '#27ae60',
    'progress':   '#8e44ad',
    'rmse':       '#c0392b',
    'kalman':     '#2980b9',
    'klt':        '#16a085',
    'lqr':        '#e74c3c',
    'mgd':        '#d35400',
    'actor':      '#e74c3c',
    'critic':     '#3498db',
    'entropy':    '#9b59b6',
    'target':     '#2980b9',
    'laser':      '#e74c3c',
    'smooth':     '#2c3e50',
}


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _latest(pattern: str, folder: Path) -> Path | None:
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None


def _load(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _smooth(s: pd.Series, w: int = 50) -> pd.Series:
    return s.rolling(window=w, min_periods=1).mean()


def _save(fig: plt.Figure, name: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {p.name}')
    return p


def _annotate_phases(ax):
    """No fixed step markers: V5 curriculum advances from measured success."""
    return None


def _actual_steps(*frames: pd.DataFrame) -> int:
    values = []
    for frame in frames:
        if not frame.empty and 'timestep' in frame.columns:
            values.append(float(frame['timestep'].max()))
    return int(max(values, default=0.0))


# ══════════════════════════════════════════════════════════════════════════════
# RL — Courbes principales
# ══════════════════════════════════════════════════════════════════════════════

def plot_rl_core(ep: pd.DataFrame, win: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if win.empty:
        print('  [SKIP] win CSV vide — RL core')
        return paths

    # 1. Reward moyen avec bande d'écart-type
    fig, ax = plt.subplots(figsize=(10, 5))
    t = win['timestep']
    r = win['reward_mean']
    std = win['reward_std'] if 'reward_std' in win.columns else pd.Series(np.zeros(len(win)))
    ax.fill_between(t, r - std, r + std, alpha=0.2, color=STYLE['reward'])
    ax.plot(t, r, color=STYLE['reward'], lw=1.0, alpha=0.5, label='Reward brut')
    ax.plot(t, _smooth(r, 5), color=STYLE['smooth'], lw=2.0, label='Moyenne lissée')
    ax.set_title('Récompense moyenne par fenêtre — Training SAC', fontweight='bold')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Reward')
    ax.legend(); ax.grid(True, alpha=0.3)
    _annotate_phases(ax)
    paths.append(_save(fig, '01_rl_reward.png', out))

    # 2. Taux de succès + progression sur même graphe
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(t, win['success_rate'] * 100, color=STYLE['success'], lw=1.5, label='Taux succès (%)')
    ax1.plot(t, _smooth(win['success_rate'], 5) * 100, color=STYLE['success'],
             lw=2.5, ls='--', label='Succès lissé')
    ax2.plot(t, win['progress_mean'] * 100, color=STYLE['progress'],
             lw=1.5, alpha=0.8, label='Progression (%)')
    ax2.plot(t, _smooth(win['progress_mean'], 5) * 100, color=STYLE['progress'],
             lw=2.5, ls='--')
    ax1.set_xlabel('Timestep'); ax1.set_ylabel('Taux de succès [%]', color=STYLE['success'])
    ax2.set_ylabel('Progression [%]', color=STYLE['progress'])
    ax1.set_title('Taux de succès et progression — Training SAC', fontweight='bold')
    ax1.set_ylim(0, 105); ax2.set_ylim(0, 105)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    _annotate_phases(ax1)
    paths.append(_save(fig, '02_rl_success_progress.png', out))

    # 3. RMSE avec percentiles P50 / P95
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, win['rmse_mean_m'] * 100, color=STYLE['rmse'], lw=1.0, alpha=0.4, label='RMSE moyen')
    if 'rmse_p50_m' in win.columns:
        ax.plot(t, win['rmse_p50_m'] * 100, color=STYLE['rmse'], lw=1.5, label='P50')
    if 'rmse_p95_m' in win.columns:
        ax.plot(t, win['rmse_p95_m'] * 100, color='#922b21', lw=1.0, ls=':', alpha=0.7, label='P95')
    ax.plot(t, _smooth(win['rmse_mean_m'], 5) * 100, color=STYLE['smooth'], lw=2.0, label='Lissé')
    ax.axhline(4.0, color='green', lw=1.5, ls='--', label='Seuil succès 4 cm')
    ax.set_title('RMSE Laser ↔ Cible [cm] — évolution training', fontweight='bold')
    ax.set_xlabel('Timestep'); ax.set_ylabel('RMSE [cm]')
    ax.legend(); ax.grid(True, alpha=0.3)
    _annotate_phases(ax)
    paths.append(_save(fig, '03_rl_rmse.png', out))

    # 4. Durée d'épisode + stagnation
    if not ep.empty and 'episode_length' in ep.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep['timestep'], ep['episode_length'], color='steelblue',
                lw=0.5, alpha=0.3, label='Durée brute')
        ax.plot(ep['timestep'], _smooth(ep['episode_length'], 30), color='steelblue',
                lw=2.0, label='Durée lissée (30 ep)')
        ax.set_title('Durée des épisodes [steps]', fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('Steps / épisode')
        ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '04_rl_episode_length.png', out))

    # 5. Curriculum level
    if not ep.empty and 'curriculum_level' in ep.columns:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ep['timestep'], ep['curriculum_level'], color='#f39c12',
                lw=1.0, drawstyle='steps-post', label='Curriculum level')
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['Lvl 0 (simple)', 'Lvl 1', 'Lvl 2 (complexe)'])
        ax.set_title('Progression du curriculum', fontweight='bold')
        ax.set_xlabel('Timestep'); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '05_rl_curriculum.png', out))

    # 6. Tableau de bord RL en une seule figure (soutenance compact)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)
    fig.suptitle(f'Tableau de bord RL — SAC Training ({_actual_steps(ep, win):,} steps réels)', fontsize=14, fontweight='bold')

    for idx, (col, title, ylabel, color) in enumerate([
        ('reward_mean',    'Reward moyen',          'Reward',    STYLE['reward']),
        ('success_rate',   'Taux de succès',         '[%]',       STYLE['success']),
        ('progress_mean',  'Progression',            '[%]',       STYLE['progress']),
        ('rmse_mean_m',    'RMSE laser↔cible',       '[m]',       STYLE['rmse']),
        ('ekf_sigma_mean_m', 'EKF σ moyen',          '[m]',       STYLE['kalman']),
        ('yoshikawa_wall_mean', 'Yoshikawa w(q)',    '[-]',       STYLE['mgd']),
    ]):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        if col in win.columns:
            scale = 100 if 'rate' in col or 'progress' in col else 1
            ax.plot(t, win[col] * scale, color=color, lw=0.8, alpha=0.4)
            ax.plot(t, _smooth(win[col], 5) * scale, color=color, lw=2.0)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        if 'rate' in col or 'progress' in col:
            ax.set_ylim(0, 105)
    paths.append(_save(fig, '06_rl_dashboard.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# SAC — Internals (losses, entropy)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sac_internals(up: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if up.empty:
        return paths

    # 7. Actor / Critic loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SAC — Pertes Actor/Critic', fontweight='bold')
    for ax, col, color, title in [
        (axes[0], 'actor_loss',  STYLE['actor'],  'Actor Loss'),
        (axes[1], 'critic_loss', STYLE['critic'], 'Critic Loss'),
    ]:
        if col in up.columns:
            ax.plot(up['timestep'], up[col], color=color, lw=0.6, alpha=0.3)
            ax.plot(up['timestep'], _smooth(up[col], 20), color=color, lw=2.0, label='Lissé')
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Timestep')
        ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3); ax.legend()
    paths.append(_save(fig, '07_sac_losses.png', out))

    # 8. Entropy coefficient
    fig, ax = plt.subplots(figsize=(10, 4))
    if 'ent_coef' in up.columns:
        ax.plot(up['timestep'], up['ent_coef'], color=STYLE['entropy'], lw=0.8, alpha=0.4)
        ax.plot(up['timestep'], _smooth(up['ent_coef'], 20), color=STYLE['entropy'],
                lw=2.0, label='α (entropie)')
    ax.set_title('SAC — Coefficient d\'entropie α (exploration/exploitation)', fontweight='bold')
    ax.set_xlabel('Timestep'); ax.set_ylabel('α'); ax.legend(); ax.grid(True, alpha=0.3)
    paths.append(_save(fig, '08_sac_entropy.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Reward — décomposition des 11 composantes
# ══════════════════════════════════════════════════════════════════════════════

def plot_reward_breakdown(ep: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if ep.empty:
        return paths

    components = {
        'Dist. ordonnée':   'reward_dist_ordered',
        'Waypoint bonus':   'reward_waypoint_bonus',
        'Completion':       'reward_completion_bonus',
        'Progress':         'reward_progress_reward',
        'Record bonus':     'reward_record_bonus',
        'Cmd penalty':      'reward_cmd_penalty',
        'Action Δ pen.':    'reward_action_delta_penalty',
        'Stagnation pen.':  'reward_stagnation_penalty',
        'Singularité pen.': 'reward_sing_penalty',
        'Vision pen.':      'reward_vision_penalty',
        'Offwall pen.':     'reward_offwall_penalty',
        'Caméra périmée':    'reward_stale_camera_penalty',
    }

    # 9. Évolution de chaque composante
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 3, hspace=0.5, wspace=0.35)
    fig.suptitle('Décomposition du Reward — composantes cumulées par épisode', fontsize=13, fontweight='bold')
    colors_pos = plt.cm.Greens(np.linspace(0.5, 0.9, 6))
    colors_neg = plt.cm.Reds(np.linspace(0.5, 0.9, 6))
    comp_items = list(components.items())
    for idx, (label, col) in enumerate(comp_items):
        if idx >= 12:
            break
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        if col in ep.columns:
            vals = ep[col]
            c = colors_pos[idx % 6] if vals.mean() >= 0 else colors_neg[idx % 6]
            ax.plot(ep['timestep'], vals, color=c, lw=0.5, alpha=0.25)
            ax.plot(ep['timestep'], _smooth(vals, 30), color=c, lw=1.8, label='Lissé')
            ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=7); ax.set_ylabel('Valeur', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    paths.append(_save(fig, '09_reward_breakdown.png', out))

    # 10. Stacked area des composantes positives vs négatives (moyennes fenêtres de 1000 ep)
    window = max(1, len(ep) // 30)
    ep_w = ep.set_index('timestep').rolling(window, min_periods=1).mean().reset_index()
    pos_cols = [c for l, c in comp_items if c in ep.columns and ep[c].mean() >= 0]
    neg_cols = [c for l, c in comp_items if c in ep.columns and ep[c].mean() < 0]
    pos_labels = [l for l, c in comp_items if c in ep.columns and ep[c].mean() >= 0]
    neg_labels = [l for l, c in comp_items if c in ep.columns and ep[c].mean() < 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Reward — Composantes positives vs négatives', fontweight='bold')
    t_ = ep_w['timestep']
    if pos_cols:
        ax1.stackplot(t_, [ep_w[c] for c in pos_cols], labels=pos_labels,
                      alpha=0.75, colors=plt.cm.Greens(np.linspace(0.4, 0.85, len(pos_cols))))
        ax1.set_ylabel('Reward positif'); ax1.legend(fontsize=7, loc='upper left')
        ax1.grid(True, alpha=0.3); ax1.set_title('Composantes positives')
    if neg_cols:
        ax2.stackplot(t_, [ep_w[c] for c in neg_cols], labels=neg_labels,
                      alpha=0.75, colors=plt.cm.Reds(np.linspace(0.4, 0.85, len(neg_cols))))
        ax2.set_ylabel('Pénalités'); ax2.legend(fontsize=7, loc='lower left')
        ax2.grid(True, alpha=0.3); ax2.set_title('Composantes négatives (pénalités)')
    ax2.set_xlabel('Timestep')
    paths.append(_save(fig, '10_reward_stacked.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Kalman EKF
# ══════════════════════════════════════════════════════════════════════════════

def plot_kalman(ep: pd.DataFrame, win: pd.DataFrame, out: Path) -> list[Path]:
    paths = []

    # 11. σ moyen EKF pendant training
    if not win.empty and 'ekf_sigma_mean_m' in win.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        t = win['timestep']
        s = win['ekf_sigma_mean_m'] * 1000
        ax.fill_between(t, 0, s, alpha=0.2, color=STYLE['kalman'])
        ax.plot(t, s, color=STYLE['kalman'], lw=1.0, alpha=0.5)
        ax.plot(t, _smooth(s, 5), color=STYLE['kalman'], lw=2.0, label='σ moyen (mm)')
        ax.axhline(6.0,  color='blue',  lw=1.0, ls='--', label='σ_FK = 6 mm')
        ax.axhline(18.0, color='red',   lw=1.0, ls=':', label='σ_cam = 18 mm')
        ax.set_title('EKF — Incertitude σ pendant l\'entraînement', fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('σ [mm]')
        ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '11_kalman_sigma.png', out))

    # 12. NIS pendant training
    if not ep.empty and 'ekf_nis' in ep.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        nis = ep['ekf_nis'].replace([np.inf, -np.inf], np.nan).dropna()
        t_nis = ep.loc[nis.index, 'timestep']
        ax.plot(t_nis, nis, color=STYLE['kalman'], lw=0.5, alpha=0.3)
        ax.plot(t_nis, _smooth(nis, 30), color=STYLE['kalman'], lw=2.0, label='NIS lissé')
        ax.axhline(0.5, color='green', lw=1.0, ls='--', label='Borne inf. (chi²/2)')
        ax.axhline(5.0, color='red',   lw=1.0, ls='--', label='Borne sup. (chi²/2)')
        ax.set_ylim(0, 10)
        ax.set_title('EKF — NIS (Normalized Innovation Squared)\n'
                     'Entre 0.5 et 5.0 → filtre cohérent', fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('NIS')
        ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '12_kalman_nis.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# KLT Caméra
# ══════════════════════════════════════════════════════════════════════════════

def plot_klt(ep: pd.DataFrame, win: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if ep.empty:
        return paths

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('KLT Caméra — Métriques pendant l\'entraînement', fontweight='bold')
    for ax, col, label, color in [
        (axes[0], 'cam_detected',      'Taux détection ligne (%)',   STYLE['klt']),
        (axes[1], 'cam_laser_visible', 'Taux laser visible (%)',     '#e67e22'),
        (axes[2], 'visual_progress',   'Progression visuelle (%)',   STYLE['progress']),
    ]:
        if col in ep.columns:
            vals = ep[col] * 100
            ax.plot(ep['timestep'], vals, color=color, lw=0.5, alpha=0.2)
            ax.plot(ep['timestep'], _smooth(vals, 30), color=color, lw=2.0, label='Lissé')
            ax.set_ylim(0, 105)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('%')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    paths.append(_save(fig, '13_klt_camera.png', out))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# LQR / Commande
# ══════════════════════════════════════════════════════════════════════════════

def plot_lqr(ep: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if ep.empty:
        return paths

    cols = [('cmd_raw_norm', 'RL brut'),
            ('cmd_lqr_norm', 'Après LQR/Riccati'),
            ('cmd_null_norm', 'Noyau Jacobien'),
            ('cmd_out_norm', 'Sortie finale')]
    colors = [STYLE['lqr'], '#e67e22', '#9b59b6', '#27ae60']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('LQR — Normes des commandes articulaires [rad/s]', fontweight='bold')

    for (col, label), color in zip(cols, colors):
        if col in ep.columns:
            ax1.plot(ep['timestep'], ep[col], color=color, lw=0.5, alpha=0.2)
            ax1.plot(ep['timestep'], _smooth(ep[col], 30), color=color, lw=2.0, label=label)

    ax1.set_ylabel('Norme ‖·‖ [rad/s]'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_title('Évolution des normes de commande')

    # Rapport cmd_lqr / cmd_raw (gain du correcteur)
    if 'cmd_lqr_norm' in ep.columns and 'cmd_raw_norm' in ep.columns:
        ratio = (ep['cmd_lqr_norm'] / ep['cmd_raw_norm'].replace(0, np.nan)).dropna()
        t_r = ep.loc[ratio.index, 'timestep']
        ax2.plot(t_r, ratio, color='#2c3e50', lw=0.5, alpha=0.2)
        ax2.plot(t_r, _smooth(ratio, 30), color='#2c3e50', lw=2.0, label='cmd_lqr/cmd_raw')
        ax2.axhline(1.0, color='gray', lw=1.0, ls='--', label='ratio=1 (pas de modif)')
    ax2.set_xlabel('Timestep'); ax2.set_ylabel('Ratio LQR/RAW')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_title('Ratio LQR/RL — impact du correcteur Riccati')

    paths.append(_save(fig, '14_lqr_commands.png', out))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# MGD — Yoshikawa / Singularités
# ══════════════════════════════════════════════════════════════════════════════

def plot_mgd(ep: pd.DataFrame, win: pd.DataFrame, out: Path) -> list[Path]:
    paths = []

    if not win.empty and 'yoshikawa_wall_mean' in win.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('MGD / Singularités — Yoshikawa et conditionnement', fontweight='bold')
        t = win['timestep']

        # Yoshikawa
        w = win['yoshikawa_wall_mean']
        ax1.fill_between(t, 0, w, alpha=0.15, color=STYLE['mgd'])
        ax1.plot(t, w, color=STYLE['mgd'], lw=1.0, alpha=0.4)
        ax1.plot(t, _smooth(w, 5), color=STYLE['mgd'], lw=2.0, label='w moyen lissé')
        ax1.axhline(0.04,  color='red',   lw=1.5, ls='--', label='W_MIN (seuil pénalité)')
        ax1.axhline(0.115, color='green', lw=1.0, ls=':', label='W_HOME (référence)')
        ax1.fill_between(t, 0, 0.04, alpha=0.1, color='red', label='Zone singulière')
        ax1.set_title('Indice de Yoshikawa w(q) = √det(JJᵀ)', fontweight='bold')
        ax1.set_xlabel('Timestep'); ax1.set_ylabel('w [-]')
        ax1.set_ylim(0, 0.2); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        # Conditionnement
        if 'cond_wall_median' in win.columns:
            c = win['cond_wall_median']
            ax2.plot(t, c, color='#8e44ad', lw=0.8, alpha=0.4)
            ax2.plot(t, _smooth(c, 5), color='#8e44ad', lw=2.0, label='κ médian lissé')
            ax2.axhline(50, color='orange', lw=1.0, ls='--', label='κ = 50 (bon)')
            ax2.axhline(200, color='red',   lw=1.0, ls=':', label='κ = 200 (mauvais)')
            ax2.set_title('Conditionnement κ = ‖J‖‖J⁺‖', fontweight='bold')
            ax2.set_xlabel('Timestep'); ax2.set_ylabel('κ [-]')
            ax2.set_yscale('log'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        paths.append(_save(fig, '15_mgd_yoshikawa.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Trajectoires snapshots (2D) — 30 figures une par une + mosaïque
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_snapshots(snap_dir: Path, out: Path) -> list[Path]:
    paths = []
    if not snap_dir.exists():
        print('  [SKIP] dossier snapshots absent')
        return paths

    laser_files  = sorted(snap_dir.glob('snap_*_laser.csv'))
    target_files = sorted(snap_dir.glob('snap_*_target.csv'))

    if not laser_files:
        print('  [SKIP] aucun snapshot laser disponible')
        return paths

    # Associer laser ↔ target par tag
    laser_by_tag  = {f.stem.replace('_laser', ''): f for f in laser_files}
    target_by_tag = {f.stem.replace('_target', ''): f for f in target_files}
    tags = sorted(laser_by_tag.keys())

    rmse_vals, step_vals = [], []

    # -- Mosaïque de tous les snapshots (max 30)
    n = min(len(tags), 30)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig_mosaic, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 3.5, nrows * 3.5))
    fig_mosaic.suptitle(
        'Évolution de la trajectoire laser — 30 snapshots (1 / 10 000 steps)',
        fontsize=13, fontweight='bold')
    ax_flat = axes.flatten() if n > 1 else [axes]

    for ax in ax_flat[n:]:
        ax.axis('off')

    for i, tag in enumerate(tags[:n]):
        ax = ax_flat[i]
        step = int(tag.replace('snap_', ''))

        target, laser = None, None
        if tag in target_by_tag:
            try:
                target = np.loadtxt(target_by_tag[tag], delimiter=',', skiprows=1)
            except Exception:
                pass
        if tag in laser_by_tag:
            try:
                laser = np.loadtxt(laser_by_tag[tag], delimiter=',', skiprows=1)
            except Exception:
                pass

        if target is not None and target.ndim == 2:
            ax.plot(target[:, 0], target[:, 1], color=STYLE['target'],
                    lw=2.0, label='Cible', zorder=2)
            ax.scatter(target[0, 0], target[0, 1],   c='#e74c3c', s=40, zorder=5)
            ax.scatter(target[-1, 0], target[-1, 1], c='#27ae60', s=40, zorder=5)

        rmse_cm = float('nan')
        if laser is not None and len(laser) > 1:
            if laser.ndim == 1:
                laser = laser.reshape(-1, 2)
            ax.plot(laser[:, 0], laser[:, 1], color=STYLE['laser'],
                    lw=1.2, alpha=0.85, label='Laser', zorder=3)
            if target is not None and len(target) > 0:
                diffs = laser[:, None, :] - target[None, :, :]
                dists = np.sqrt((diffs ** 2).sum(axis=2))
                rmse_cm = float(np.sqrt(np.mean(np.min(dists, axis=1) ** 2))) * 100
                rmse_vals.append(rmse_cm)
                step_vals.append(step)

        ax.set_title(f'Step {step:,}\nRMSE={rmse_cm:.1f}cm' if not np.isnan(rmse_cm)
                     else f'Step {step:,}', fontsize=7, fontweight='bold')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    paths.append(_save(fig_mosaic, '16_trajectory_mosaic.png', out))

    # -- Évolution RMSE des snapshots
    if rmse_vals:
        fig, ax = plt.subplots(figsize=(10, 5))
        color_map = plt.cm.RdYlGn_r(Normalize(0, max(rmse_vals))(rmse_vals))
        sc = ax.scatter(step_vals, rmse_vals, c=rmse_vals, cmap='RdYlGn_r',
                        s=80, zorder=3, label='RMSE snapshot')
        ax.plot(step_vals, rmse_vals, color='gray', lw=1.0, alpha=0.5, zorder=2)
        ax.axhline(4.0, color='green', lw=1.5, ls='--', label='Seuil succès 4 cm')
        plt.colorbar(sc, ax=ax, label='RMSE [cm]')
        ax.set_title('RMSE trajectoire laser ↔ cible — évolution par snapshot',
                     fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('RMSE [cm]')
        ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '17_trajectory_rmse_evolution.png', out))

    # -- Superposition début/milieu/fin (3 comparaisons détaillées)
    selected_idx = [0, n // 2, n - 1]
    selected_tags = [tags[i] for i in selected_idx if i < len(tags)]
    fig, axes3 = plt.subplots(1, len(selected_tags), figsize=(6 * len(selected_tags), 6))
    if len(selected_tags) == 1:
        axes3 = [axes3]
    fig.suptitle('Comparaison Début / Milieu / Fin du training', fontweight='bold', fontsize=12)

    for ax, tag in zip(axes3, selected_tags):
        step = int(tag.replace('snap_', ''))
        target, laser = None, None
        if tag in target_by_tag:
            try:
                target = np.loadtxt(target_by_tag[tag], delimiter=',', skiprows=1)
            except Exception:
                pass
        if tag in laser_by_tag:
            try:
                laser = np.loadtxt(laser_by_tag[tag], delimiter=',', skiprows=1)
            except Exception:
                pass

        if target is not None and target.ndim == 2:
            ax.plot(target[:, 0], target[:, 1], color=STYLE['target'], lw=2.5,
                    label='Cible (bleue)', zorder=2)
            ax.scatter(target[0, 0], target[0, 1], c='#e74c3c', s=80, zorder=5,
                       label='Départ', marker='^')
            ax.scatter(target[-1, 0], target[-1, 1], c='#27ae60', s=80, zorder=5,
                       label='Arrivée', marker='*')

        rmse_cm = float('nan')
        if laser is not None and len(laser) > 0:
            if laser.ndim == 1:
                laser = laser.reshape(-1, 2)
            ax.plot(laser[:, 0], laser[:, 1], color=STYLE['laser'], lw=2.0,
                    alpha=0.85, label='Laser RL', zorder=3)
            if target is not None and len(target) > 0:
                diffs = laser[:, None, :] - target[None, :, :]
                dists = np.sqrt((diffs ** 2).sum(axis=2))
                rmse_cm = float(np.sqrt(np.mean(np.min(dists, axis=1) ** 2))) * 100

        title = f'Step {step:,}'
        if not np.isnan(rmse_cm):
            title += f'\nRMSE = {rmse_cm:.1f} cm'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    paths.append(_save(fig, '18_trajectory_before_after.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Distributions et analyses statistiques
# ══════════════════════════════════════════════════════════════════════════════

def plot_distributions(ep: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if ep.empty:
        return paths

    # 19. Histogramme RMSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Distributions statistiques — RMSE et reward', fontweight='bold')

    if 'ep_rmse_m' in ep.columns:
        rmse_cm = ep['ep_rmse_m'].dropna() * 100
        # Comparer début vs fin (premier et dernier tiers)
        third = len(rmse_cm) // 3
        axes[0].hist(rmse_cm.iloc[:third], bins=40, alpha=0.6, color='#e74c3c',
                     density=True, label='Début training')
        axes[0].hist(rmse_cm.iloc[2*third:], bins=40, alpha=0.6, color='#27ae60',
                     density=True, label='Fin training')
        axes[0].axvline(4.0, color='black', lw=1.5, ls='--', label='Seuil 4 cm')
        axes[0].set_title('Distribution RMSE — début vs fin')
        axes[0].set_xlabel('RMSE [cm]'); axes[0].set_ylabel('Densité')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if 'episode_reward' in ep.columns:
        rew = ep['episode_reward'].dropna()
        third = len(rew) // 3
        axes[1].hist(rew.iloc[:third], bins=40, alpha=0.6, color='#e74c3c',
                     density=True, label='Début training')
        axes[1].hist(rew.iloc[2*third:], bins=40, alpha=0.6, color='#27ae60',
                     density=True, label='Fin training')
        axes[1].set_title('Distribution Reward')
        axes[1].set_xlabel('Reward'); axes[1].set_ylabel('Densité')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

    paths.append(_save(fig, '19_distributions.png', out))

    # 20. Scatter reward vs RMSE coloré par succès et progression
    if 'ep_rmse_m' in ep.columns and 'episode_reward' in ep.columns:
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(ep['ep_rmse_m'] * 100, ep['episode_reward'],
                        c=ep['progress'] if 'progress' in ep.columns else 'blue',
                        cmap='viridis', alpha=0.4, s=12)
        plt.colorbar(sc, ax=ax, label='Progression [0-1]')
        ax.axvline(4.0, color='green', lw=1.5, ls='--', label='Seuil RMSE 4 cm')
        ax.set_title('Reward vs RMSE — coloré par progression', fontweight='bold')
        ax.set_xlabel('RMSE [cm]'); ax.set_ylabel('Reward')
        ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '20_scatter_reward_rmse.png', out))

    # 21. Succès par niveau de curriculum
    if 'curriculum_level' in ep.columns and 'is_success' in ep.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for level in [0, 1, 2]:
            mask = ep['curriculum_level'] == level
            if mask.sum() < 5:
                continue
            sub = ep[mask].copy()
            w = max(1, len(sub) // 20)
            ax.plot(sub['timestep'],
                    sub['success'].rolling(w, min_periods=1).mean() * 100,
                    lw=2.0, label=f'Curriculum {level}')
        ax.set_title('Taux de succès par niveau de curriculum', fontweight='bold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('Succès [%]')
        ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, '21_success_by_curriculum.png', out))

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 3D Trajectory (plotly html)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_trajectory(snap_dir: Path, out: Path) -> list[Path]:
    paths = []
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  [SKIP] plotly non installé — pas de 3D')
        return paths

    laser_files = sorted(snap_dir.glob('snap_*_laser.csv'))
    target_files = sorted(snap_dir.glob('snap_*_target.csv'))

    if not laser_files:
        return paths

    laser_by_tag  = {f.stem.replace('_laser', ''): f for f in laser_files}
    target_by_tag = {f.stem.replace('_target', ''): f for f in target_files}
    tags = sorted(laser_by_tag.keys())

    fig3d = go.Figure()
    n = min(len(tags), 10)  # max 10 pour lisibilité
    color_scale = [f'rgb({int(255*(1-i/n))},{int(255*i/n)},50)' for i in range(n)]

    for i, tag in enumerate(tags[:n]):
        step = int(tag.replace('snap_', ''))
        if tag in laser_by_tag:
            try:
                laser = np.loadtxt(laser_by_tag[tag], delimiter=',', skiprows=1)
                if laser.ndim == 2 and len(laser) > 1:
                    fig3d.add_trace(go.Scatter3d(
                        x=[step] * len(laser), y=laser[:, 0], z=laser[:, 1],
                        mode='lines', name=f'Laser step {step:,}',
                        line=dict(color=color_scale[i], width=3)
                    ))
            except Exception:
                pass

    if tag in target_by_tag:
        try:
            target = np.loadtxt(target_by_tag[tags[-1]], delimiter=',', skiprows=1)
            if target.ndim == 2:
                fig3d.add_trace(go.Scatter3d(
                    x=[tags[-1]] * len(target), y=target[:, 0], z=target[:, 1],
                    mode='lines', name='Cible',
                    line=dict(color='blue', width=4, dash='dash')
                ))
        except Exception:
            pass

    fig3d.update_layout(
        title='Évolution 3D des trajectoires laser (step × y × z)',
        scene=dict(xaxis_title='Step', yaxis_title='y (m)', zaxis_title='z (m)'),
        height=700,
    )
    out_path = out / '22_trajectory_3d.html'
    out.mkdir(parents=True, exist_ok=True)
    fig3d.write_html(str(out_path))
    print(f'  [OK] {out_path.name}')
    paths.append(out_path)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Résumé global
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(ep: pd.DataFrame, win: pd.DataFrame, out: Path) -> list[Path]:
    """Une seule figure grand format : la plus importante pour la soutenance."""
    paths = []
    if win.empty and ep.empty:
        return paths

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f'UR7e Line Follower — Résumé SAC ({_actual_steps(ep, win):,} steps réels)',
        fontsize=16, fontweight='bold', y=0.98)

    metrics_win = [
        ('reward_mean', 'Reward moyen', STYLE['reward'], '[0]'),
        ('success_rate', 'Taux de succès [%]', STYLE['success'], '[1]'),
        ('progress_mean', 'Progression [%]', STYLE['progress'], '[2]'),
        ('rmse_mean_m', 'RMSE [cm]', STYLE['rmse'], '[3]'),
        ('ekf_sigma_mean_m', 'EKF σ [mm]', STYLE['kalman'], '[4]'),
        ('yoshikawa_wall_mean', 'Yoshikawa w', STYLE['mgd'], '[5]'),
    ]
    metrics_ep = [
        ('cmd_out_norm', '‖Commande‖ [rad/s]', STYLE['lqr'], '[6]'),
        ('cam_detected', 'KLT détection [%]', STYLE['klt'], '[7]'),
    ]

    t_win = win['timestep'] if not win.empty else pd.Series(dtype=float)

    for idx, (col, title, color, tag) in enumerate(metrics_win):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        if not win.empty and col in win.columns:
            scale = 100 if 'rate' in col or 'progress' in col else \
                    (1000 if 'sigma' in col else (100 if 'rmse' in col else 1))
            v = win[col] * scale
            ax.plot(t_win, v, color=color, lw=0.7, alpha=0.3)
            ax.plot(t_win, _smooth(v, 5), color=color, lw=2.0)
        ax.set_title(f'{tag} {title}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=7); ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx, (col, title, color, tag) in enumerate(metrics_ep):
        ax = fig.add_subplot(gs[2, idx * 2:(idx + 1) * 2])
        if not ep.empty and col in ep.columns:
            scale = 100 if 'detect' in col or 'visible' in col else 1
            v = ep[col] * scale
            ax.plot(ep['timestep'], v, color=color, lw=0.5, alpha=0.2)
            ax.plot(ep['timestep'], _smooth(v, 30), color=color, lw=2.0)
        ax.set_title(f'{tag} {title}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=7); ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    paths.append(_save(fig, '00_SUMMARY.png', out))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Générateur de courbes post-training pour soutenance UR7e')
    parser.add_argument('--metrics', default=None,
                        help='Dossier metrics ou dossier de run; défaut: dernier run isolé')
    parser.add_argument('--out', default=str(OUT_DIR),
                        help='Dossier de sortie des figures')
    args = parser.parse_args()

    metrics = _resolve_metrics_dir(args.metrics)
    out     = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    snap_dir = metrics / 'trajectory_snapshots'

    print(f'\n[plot_training] Lecture métriques : {metrics}')
    print(f'[plot_training] Sortie figures    : {out}\n')

    ep  = _load(_latest('train_episodes_*.csv', metrics))
    win = _load(_latest('train_windows_*.csv',  metrics))
    up  = _load(_latest('train_updates_*.csv',  metrics))

    if ep.empty and win.empty:
        print('[ERREUR] Aucun CSV trouvé. Vérifiez --metrics.')
        return

    print(f'Episodes CSV  : {len(ep)} lignes')
    print(f'Windows CSV   : {len(win)} lignes')
    print(f'Updates CSV   : {len(up)} lignes\n')

    all_paths = []
    print('── RL Core ──────────────────')
    all_paths += plot_rl_core(ep, win, out)

    print('── SAC Internals ────────────')
    all_paths += plot_sac_internals(up, out)

    print('── Reward Breakdown ─────────')
    all_paths += plot_reward_breakdown(ep, out)

    print('── Kalman EKF ───────────────')
    all_paths += plot_kalman(ep, win, out)

    print('── KLT Caméra ───────────────')
    all_paths += plot_klt(ep, win, out)

    print('── LQR / Commande ───────────')
    all_paths += plot_lqr(ep, out)

    print('── MGD / Yoshikawa ──────────')
    all_paths += plot_mgd(ep, win, out)

    print('── Trajectory Snapshots ─────')
    all_paths += plot_trajectory_snapshots(snap_dir, out)

    print('── Distributions ────────────')
    all_paths += plot_distributions(ep, out)

    print('── 3D (Plotly) ──────────────')
    all_paths += plot_3d_trajectory(snap_dir, out)

    print('── Résumé global ────────────')
    all_paths += plot_summary(ep, win, out)

    print(f'\n[plot_training] {len(all_paths)} figures générées dans {out}/')
    print('[plot_training] Fichier principal : 00_SUMMARY.png')


if __name__ == '__main__':
    main()
