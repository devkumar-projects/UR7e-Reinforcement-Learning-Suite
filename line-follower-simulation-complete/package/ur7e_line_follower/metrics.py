"""
Analyse et visualisation complète des métriques d'entraînement RL.

Génère toutes les courbes standard en ML robotique et deep learning :
  1.  Courbe de récompense (rolling mean + IC 95 %)
  2.  Taux de succès (complétion ligne)
  3.  Progression moyenne le long de la ligne
  4.  Longueur d'épisode
  5.  Écart moyen à la ligne (cm) — par épisode
  6.  Écart max à la ligne (cm)
  7.  RMSE de l'écart (moindres carrés)
  8.  Histogramme de la distribution des écarts
  9.  Régression linéaire (moindres carrés) sur l'écart moyen
  10. Actor loss
  11. Critic loss
  12. Entropie (ent_coef)
  13. Learning rate
  14. Q-value estimate (critic loss proxy)
  15. Rapport signal/bruit de la récompense
  16. Corrélation écart ↔ progression
  17. Heatmap: progrès vs écart (matrice densité)
  18. Courbe d'apprentissage cumulée (reward cumulé par étape)
  19. Comparaison multi-runs (si plusieurs CSV disponibles)
  20. Dashboard récapitulatif (4-panel)

Usage :
    ros2 run ur7e_line_follower metrics              # dernière run
    ros2 run ur7e_line_follower metrics -- --run all # toutes les runs
    python3 -m ur7e_line_follower.metrics            # idem

Sorties : ~/.ros/ur7e_line_follower/plots/
"""
import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as sp_stats
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA_DIR   = Path.home() / '.ros' / 'ur7e_line_follower'
LEGACY_METRICS_DIR = DATA_DIR / 'metrics'
PLOTS_DIR = DATA_DIR / 'plots'


def _resolve_metrics_dir(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        return p / 'metrics' if (p / 'metrics').is_dir() else p
    latest = DATA_DIR / 'latest_run.txt'
    if latest.exists():
        run_dir = Path(latest.read_text().strip()).expanduser()
        if (run_dir / 'metrics').is_dir():
            return run_dir / 'metrics'
    return LEGACY_METRICS_DIR


def _normalise_episode_keys(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    aliases = {
        'episode_reward': ('reward',),
        'ep_rmse_m': ('ep_rmse', 'deviation_rmse_m'),
        'yoshikawa_wall': ('yoshikawa_w',),
    }
    for source, targets in aliases.items():
        for target in targets:
            if source in data and target not in data:
                data[target] = data[source]
    return data


# ── CSV loading ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v not in ('', 'nan') else float('nan')
                         for k, v in row.items()})
    if not rows:
        return {}
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def load_latest_run(metrics_dir: Path):
    ep_files = sorted(metrics_dir.glob('train_episodes_*.csv')) or sorted(metrics_dir.glob('episodes_*.csv'))
    up_files = sorted(metrics_dir.glob('train_updates_*.csv')) or sorted(metrics_dir.glob('updates_*.csv'))
    if not ep_files:
        raise FileNotFoundError(f"Aucun CSV épisodes dans {metrics_dir}")
    ep = _normalise_episode_keys(_load_csv(ep_files[-1]))
    up = _load_csv(up_files[-1]) if up_files else {}
    print(f"[metrics] Episodes : {ep_files[-1].name}")
    if up_files:
        print(f"[metrics] Updates  : {up_files[-1].name}")
    return ep, up


def load_all_runs(metrics_dir: Path):
    ep_files = sorted(metrics_dir.glob('train_episodes_*.csv')) or sorted(metrics_dir.glob('episodes_*.csv'))
    runs = []
    for f in ep_files:
        tag = f.stem.replace('episodes_', '')
        runs.append((tag, _normalise_episode_keys(_load_csv(f))))
    return runs


# ── Smoothing ──────────────────────────────────────────────────────────────────

def _smooth(y: np.ndarray, window: int = 15) -> np.ndarray:
    if len(y) < window:
        return y.copy()
    if HAS_SCIPY:
        w = min(window, len(y) // 2 * 2 - 1)
        if w >= 3:
            return savgol_filter(y, w, 2)
    k = np.ones(window) / window
    return np.convolve(y, k, mode='same')


def _rolling_ci(y: np.ndarray, window: int = 20, ci: float = 0.95):
    """Rolling mean ± confidence interval."""
    mean, lo, hi = [], [], []
    z = sp_stats.norm.ppf((1 + ci) / 2) if HAS_SCIPY else 1.96
    for i in range(len(y)):
        w = y[max(0, i - window + 1): i + 1]
        m = np.nanmean(w)
        s = np.nanstd(w) / np.sqrt(len(w))
        mean.append(m)
        lo.append(m - z * s)
        hi.append(m + z * s)
    return np.array(mean), np.array(lo), np.array(hi)


# ── Plot helpers ──────────────────────────────────────────────────────────────

STYLE = {
    'reward':     '#2196F3',
    'success':    '#4CAF50',
    'deviation':  '#FF5722',
    'progress':   '#9C27B0',
    'loss_actor': '#F44336',
    'loss_critic':'#FF9800',
    'entropy':    '#00BCD4',
    'lr':         '#607D8B',
    'ci_alpha':   0.18,
}

def _setup_ax(ax, title: str, xlabel: str = 'Timestep', ylabel: str = ''):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(labelsize=8)


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_reward(ep: dict, ax=None, save_dir: Path = None):
    fig, created = (None, False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x, y = ep['timestep'], ep['reward']
    mean, lo, hi = _rolling_ci(y)
    ax.fill_between(x, lo, hi, alpha=STYLE['ci_alpha'], color=STYLE['reward'])
    ax.plot(x, y, alpha=0.3, color=STYLE['reward'], lw=0.8)
    ax.plot(x, mean, color=STYLE['reward'], lw=2, label='Récompense (rolling mean)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    _setup_ax(ax, 'Récompense par épisode', ylabel='Récompense cumulée')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '01_reward.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 01_reward.png")


def plot_success(ep: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x, y = ep['timestep'], ep['success']
    smooth = _smooth(y * 100, window=20)
    ax.plot(x, smooth, color=STYLE['success'], lw=2, label='Taux de succès (%)')
    ax.scatter(x, y * 100, s=4, alpha=0.2, color=STYLE['success'])
    ax.set_ylim(-5, 105)
    _setup_ax(ax, 'Taux de complétion de la ligne', ylabel='Succès (%)')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '02_success.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 02_success.png")


def plot_progress(ep: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x, y = ep['timestep'], ep['progress'] * 100
    smooth = _smooth(y, window=20)
    ax.plot(x, smooth, color=STYLE['progress'], lw=2, label='Progression (%)')
    ax.fill_between(x, 0, smooth, alpha=0.12, color=STYLE['progress'])
    ax.set_ylim(-2, 105)
    _setup_ax(ax, 'Progression le long de la ligne', ylabel='% de la ligne parcourue')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '03_progress.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 03_progress.png")


def plot_deviation(ep: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x = ep['timestep']
    mean_d = ep['deviation_mean_m'] * 100
    max_d  = ep['deviation_max_m'] * 100
    rmse_d = ep['deviation_rmse_m'] * 100

    ax.plot(x, _smooth(mean_d, 15), color=STYLE['deviation'], lw=2, label='Écart moyen (cm)')
    ax.plot(x, _smooth(max_d,  15), color=STYLE['deviation'], lw=1.2,
            linestyle='--', alpha=0.7, label='Écart max (cm)')
    ax.plot(x, _smooth(rmse_d, 15), color='#795548', lw=1.5,
            linestyle=':', label='RMSE (cm)')
    _setup_ax(ax, 'Écart laser ↔ ligne cible', ylabel='Écart (cm)')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '04_deviation.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 04_deviation.png")


def plot_least_squares(ep: dict, ax=None, save_dir: Path = None):
    """Régression linéaire (moindres carrés) sur l'écart moyen — montre la tendance."""
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x = ep['timestep']
    y = ep['deviation_mean_m'] * 100

    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    if len(xv) >= 2:
        if HAS_SCIPY:
            slope, intercept, r, p, se = sp_stats.linregress(xv, yv)
        else:
            coeffs = np.polyfit(xv, yv, 1)
            slope, intercept = coeffs[0], coeffs[1]
            r = np.corrcoef(xv, yv)[0, 1]
            p, se = float('nan'), float('nan')
        y_fit = slope * xv + intercept
        ax.scatter(xv, yv, s=6, alpha=0.25, color=STYLE['deviation'], label='Données brutes')
        ax.plot(xv, y_fit, 'r-', lw=2.5,
                label=f'Régression MCO: {slope*1e4:.2f}×10⁻⁴·step + {intercept:.1f}  (r={r:.3f})')
        ax.plot(x, _smooth(y, 20), color='#333', lw=1.2, alpha=0.7, linestyle='--',
                label='Lissé')
    _setup_ax(ax, 'Régression moindres carrés — écart à la ligne', ylabel='Écart moyen (cm)')
    ax.legend(fontsize=7.5)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '05_least_squares.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 05_least_squares.png")


def plot_deviation_histogram(ep: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4)); created = True
    y = ep['deviation_mean_m'] * 100
    y = y[~np.isnan(y)]
    ax.hist(y, bins=30, color=STYLE['deviation'], alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(y), color='red', lw=2, linestyle='--',
               label=f'Moyenne : {np.mean(y):.1f} cm')
    ax.axvline(np.median(y), color='orange', lw=2, linestyle=':',
               label=f'Médiane : {np.median(y):.1f} cm')
    _setup_ax(ax, 'Distribution des écarts à la ligne', xlabel='Écart moyen par épisode (cm)',
              ylabel='Fréquence')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '06_deviation_histogram.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 06_deviation_histogram.png")


def plot_actor_critic_loss(up: dict, ax1=None, ax2=None, save_dir: Path = None):
    fig, created = None, False
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)); created = True
    if 'timestep' not in up:
        return
    x = up['timestep']
    for ax, key, color, title in [
        (ax1, 'actor_loss',  STYLE['loss_actor'],  'Actor Loss'),
        (ax2, 'critic_loss', STYLE['loss_critic'], 'Critic Loss'),
    ]:
        if key in up:
            y = up[key]
            ax.plot(x, y, alpha=0.3, color=color, lw=0.8)
            ax.plot(x, _smooth(y, 20), color=color, lw=2, label=title)
            _setup_ax(ax, title, ylabel='Loss')
            ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '07_losses.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 07_losses.png")


def plot_entropy(up: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    if 'ent_coef' not in up:
        return
    x, y = up['timestep'], up['ent_coef']
    ax.plot(x, y, color=STYLE['entropy'], lw=2, label='Entropie (ent_coef)')
    ax.fill_between(x, 0, y, alpha=0.12, color=STYLE['entropy'])
    _setup_ax(ax, "Coefficient d'entropie SAC", ylabel='ent_coef')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '08_entropy.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 08_entropy.png")


def plot_learning_rate(up: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    if 'learning_rate' not in up:
        return
    x, y = up['timestep'], up['learning_rate']
    ax.plot(x, y, color=STYLE['lr'], lw=2)
    ax.set_yscale('log')
    _setup_ax(ax, 'Learning Rate', ylabel='LR (log)')
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '09_learning_rate.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 09_learning_rate.png")


def plot_cumulative_reward(ep: dict, ax=None, save_dir: Path = None):
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    x, y = ep['timestep'], np.nancumsum(ep['reward'])
    ax.plot(x, y, color='#3F51B5', lw=2)
    ax.fill_between(x, 0, y, alpha=0.1, color='#3F51B5')
    _setup_ax(ax, 'Récompense cumulée', ylabel='∑ reward')
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '10_cumulative_reward.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 10_cumulative_reward.png")


def plot_correlation_deviation_progress(ep: dict, ax=None, save_dir: Path = None):
    """Scatter: deviation vs progression — montre si le robot hésite sur les zones difficiles."""
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5)); created = True
    x = ep['progress'] * 100
    y = ep['deviation_mean_m'] * 100
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    ax.scatter(xv, yv, s=10, alpha=0.4, color=STYLE['deviation'])
    if HAS_SCIPY and len(xv) >= 2:
        r, p = sp_stats.pearsonr(xv, yv)
        ax.set_title(f'Écart vs Progression  (r={r:.3f}, p={p:.3g})',
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title('Écart vs Progression', fontsize=11, fontweight='bold')
    ax.set_xlabel('Progression le long de la ligne (%)', fontsize=9)
    ax.set_ylabel('Écart moyen à la ligne (cm)', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '11_deviation_vs_progress.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 11_deviation_vs_progress.png")


def plot_density_heatmap(ep: dict, ax=None, save_dir: Path = None):
    """2D density: progression × écart — zones d'apprentissage lentes."""
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5)); created = True
    x = ep['progress'] * 100
    y = ep['deviation_mean_m'] * 100
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    if len(xv) < 10:
        return
    h, xedges, yedges = np.histogram2d(xv, yv, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, extent=extent, origin='lower', aspect='auto',
                   cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Nb épisodes')
    ax.set_xlabel('Progression (%)', fontsize=9)
    ax.set_ylabel('Écart (cm)', fontsize=9)
    ax.set_title('Densité: Progression × Écart', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '12_density_heatmap.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 12_density_heatmap.png")


def plot_snr(ep: dict, ax=None, save_dir: Path = None):
    """Signal-to-Noise Ratio de la récompense — mesure la cohérence de l'apprentissage."""
    fig, created = None, False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)); created = True
    y = ep['reward']
    window = 30
    snr = []
    for i in range(len(y)):
        w = y[max(0, i - window + 1): i + 1]
        std = np.nanstd(w)
        snr.append(np.nanmean(w) / (std + 1e-8))
    snr = np.array(snr)
    ax.plot(ep['timestep'], snr, color='#009688', lw=1.5, label='SNR reward')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    _setup_ax(ax, 'Signal/Bruit de la récompense (stabilité)', ylabel='SNR')
    ax.legend(fontsize=8)
    if created and save_dir:
        fig.tight_layout()
        fig.savefig(save_dir / '13_snr.png', dpi=150)
        plt.close(fig)
        print(f"[metrics] → 13_snr.png")


def plot_multi_run_comparison(runs: list, metric: str = 'reward', save_dir: Path = None):
    """Compare plusieurs runs sur la même métrique."""
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap('tab10')
    for i, (tag, ep) in enumerate(runs):
        if metric not in ep:
            continue
        x, y = ep['timestep'], ep[metric]
        smooth = _smooth(y if metric != 'deviation_mean_m' else y * 100, 20)
        ax.plot(x, smooth, lw=2, color=cmap(i), label=tag)
    label_map = {
        'reward': 'Récompense', 'success': 'Succès (%)',
        'deviation_mean_m': 'Écart moyen (cm)', 'progress': 'Progression (%)',
    }
    title_metric = label_map.get(metric, metric)
    _setup_ax(ax, f'Comparaison multi-runs — {title_metric}', ylabel=title_metric)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_dir:
        fname = f'14_multirun_{metric}.png'
        fig.savefig(save_dir / fname, dpi=150)
        plt.close(fig)
        print(f"[metrics] → {fname}")


def plot_dashboard(ep: dict, up: dict, save_dir: Path = None):
    """
    Dashboard récapitulatif 4-panel :
      [Récompense | Succès]
      [Écart MCO  | Actor/Critic Loss]
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    ax_rew = fig.add_subplot(gs[0, 0])
    ax_suc = fig.add_subplot(gs[0, 1])
    ax_dev = fig.add_subplot(gs[1, 0])
    ax_los = fig.add_subplot(gs[1, 1])

    plot_reward(ep, ax=ax_rew)
    plot_success(ep, ax=ax_suc)
    plot_least_squares(ep, ax=ax_dev)

    if up and 'actor_loss' in up:
        x = up['timestep']
        ax_los.plot(x, _smooth(up['actor_loss'],  20),
                    color=STYLE['loss_actor'],  lw=2, label='Actor loss')
        ax_los.plot(x, _smooth(up['critic_loss'], 20),
                    color=STYLE['loss_critic'], lw=2, label='Critic loss')
        _setup_ax(ax_los, 'Losses', ylabel='Loss')
        ax_los.legend(fontsize=8)
    else:
        plot_progress(ep, ax=ax_los)

    n_ep = int(ep['episode'][-1]) if 'episode' in ep else len(ep['timestep'])
    ts   = int(ep['timestep'][-1]) if len(ep['timestep']) > 0 else 0
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(
        f'UR7e Line Follower — Dashboard RL  ({ts:,} steps | {n_ep} épisodes | {run_date})',
        fontsize=13, fontweight='bold', y=0.98,
    )

    if save_dir:
        fig.savefig(save_dir / '00_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[metrics] → 00_dashboard.png")


# ── Stats résumé ───────────────────────────────────────────────────────────────

def print_summary(ep: dict, up: dict):
    print("\n" + "═" * 58)
    print("  RÉSUMÉ MÉTRIQUES D'ENTRAÎNEMENT")
    print("═" * 58)
    if 'timestep' in ep:
        print(f"  Timesteps totaux    : {int(ep['timestep'][-1]):>10,}")
    if 'episode' in ep:
        print(f"  Épisodes totaux     : {int(ep['episode'][-1]):>10,}")

    last_n = min(50, len(ep.get('reward', [])))
    if last_n > 0:
        r = ep['reward'][-last_n:]
        print(f"\n  [ Derniers {last_n} épisodes ]")
        print(f"  Récompense moy      : {np.nanmean(r):>+10.2f}")
        print(f"  Récompense std      : {np.nanstd(r):>10.2f}")

    if 'success' in ep:
        s = ep['success'][-last_n:]
        print(f"  Taux de succès      : {np.nanmean(s)*100:>9.1f} %")

    if 'progress' in ep:
        p = ep['progress'][-last_n:]
        print(f"  Progression moy     : {np.nanmean(p)*100:>9.1f} %")

    if 'deviation_mean_m' in ep:
        d = ep['deviation_mean_m'][-last_n:] * 100
        rmse = float(np.sqrt(np.nanmean(d ** 2)))
        print(f"  Écart moy à la ligne: {np.nanmean(d):>9.2f} cm")
        print(f"  RMSE écart          : {rmse:>9.2f} cm")
        print(f"  Écart max moy       : {np.nanmean(ep['deviation_max_m'][-last_n:]*100):>9.2f} cm")

        # Régression MCO sur toutes les données
        all_d = ep['deviation_mean_m'] * 100
        all_t = ep['timestep']
        valid = ~np.isnan(all_d)
        if valid.sum() >= 2:
            if HAS_SCIPY:
                slope, intercept, r_val, _, _ = sp_stats.linregress(all_t[valid], all_d[valid])
            else:
                c = np.polyfit(all_t[valid], all_d[valid], 1)
                slope, intercept, r_val = c[0], c[1], np.corrcoef(all_t[valid], all_d[valid])[0, 1]
            trend = "↓ améliore" if slope < 0 else "↑ dégrade"
            print(f"\n  [ Régression MCO écart ]")
            print(f"  Pente               : {slope*1000:.4f} cm/1000 steps  ({trend})")
            print(f"  Ordonnée à l'origine: {intercept:.2f} cm")
            print(f"  Corrélation (r)     : {r_val:.4f}")

    if up and 'actor_loss' in up:
        last_u = min(50, len(up['actor_loss']))
        print(f"\n  [ Updates réseau (derniers {last_u}) ]")
        print(f"  Actor loss moy      : {np.nanmean(up['actor_loss'][-last_u:]):>10.4f}")
        print(f"  Critic loss moy     : {np.nanmean(up['critic_loss'][-last_u:]):>10.4f}")
        if 'ent_coef' in up:
            print(f"  Entropie (ent_coef) : {np.nanmean(up['ent_coef'][-last_u:]):>10.6f}")

    print("═" * 58 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not HAS_MPL:
        print("[metrics] ERREUR : matplotlib non installé. pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Génère les courbes d\'entraînement RL')
    parser.add_argument('--run', default='latest',
                        help='"latest" (défaut) ou "all" pour comparer toutes les runs')
    parser.add_argument('--out', default=str(PLOTS_DIR),
                        help='Dossier de sortie pour les graphiques')
    parser.add_argument('--metrics', default=None,
                        help='Dossier metrics ou dossier de run; défaut: dernier run')
    args = parser.parse_args()

    metrics_dir = _resolve_metrics_dir(args.metrics)
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[metrics] Sortie : {save_dir}")

    if args.run == 'all':
        runs = load_all_runs(metrics_dir)
        if not runs:
            print(f"[metrics] Aucune run trouvée dans {metrics_dir}")
            return
        ep, up = runs[-1][1], {}
        plot_multi_run_comparison(runs, 'reward', save_dir)
        plot_multi_run_comparison(runs, 'deviation_mean_m', save_dir)
        plot_multi_run_comparison(runs, 'success', save_dir)
    else:
        ep, up = load_latest_run(metrics_dir)

    print_summary(ep, up)

    # Generate all individual plots
    plot_dashboard(ep, up, save_dir)
    plot_reward(ep, save_dir=save_dir)
    plot_success(ep, save_dir=save_dir)
    plot_progress(ep, save_dir=save_dir)
    plot_deviation(ep, save_dir=save_dir)
    plot_least_squares(ep, save_dir=save_dir)
    plot_deviation_histogram(ep, save_dir=save_dir)
    plot_actor_critic_loss(up, save_dir=save_dir)
    plot_entropy(up, save_dir=save_dir)
    plot_learning_rate(up, save_dir=save_dir)
    plot_cumulative_reward(ep, save_dir=save_dir)
    plot_correlation_deviation_progress(ep, save_dir=save_dir)
    plot_density_heatmap(ep, save_dir=save_dir)
    plot_snr(ep, save_dir=save_dir)

    print(f"\n[metrics] {len(list(save_dir.glob('*.png')))} graphiques générés dans {save_dir}")


if __name__ == '__main__':
    main()
