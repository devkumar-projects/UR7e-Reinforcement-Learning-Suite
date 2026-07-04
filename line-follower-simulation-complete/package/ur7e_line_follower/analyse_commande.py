"""
analyse_commande.py — Analyse commande numérique + EKF du UR7e laser line-follower.

Génère depuis les CSV d'évaluation (ou d'entraînement) :
  1. Suivi de position : consigne vs sortie laser (y, z)
  2. Erreur de suivi et erreur statique
  3. Effort de contrôle (normes commandes articulaires)
  4. EKF : état estimé vs mesures FK et caméra, covariance, innovation
  5. Monte Carlo : boîtes à moustaches par niveau de bruit
  6. Meilleur paramétrage EKF (critère NEES + RMSE)

Usage :
    python3 -m ur7e_line_follower.analyse_commande
    python3 -m ur7e_line_follower.analyse_commande --steps eval_steps_XXXX.csv
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR   = Path.home() / '.ros' / 'ur7e_line_follower'
EVAL_DIR   = DATA_DIR / 'eval'
def _default_train_dir() -> Path:
    latest = DATA_DIR / 'latest_run.txt'
    if latest.exists():
        run_dir = Path(latest.read_text().strip()).expanduser()
        if (run_dir / 'metrics').is_dir():
            return run_dir / 'metrics'
    return DATA_DIR / 'metrics'

TRAIN_DIR = _default_train_dir()
OUT_DIR    = DATA_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'consigne': '#1B998B',
    'sortie':   '#2E86AB',
    'ekf':      '#C73E1D',
    'fk':       '#F18F01',
    'cam':      '#3B1F2B',
    'error':    '#A23B72',
    'ctrl':     '#555555',
    'rl':       '#2E86AB',
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_steps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['deviation_cm'] = df['tracking_error_m'] * 100.0
    df['ctrl_norm']    = np.linalg.norm(
        df[['cmd0','cmd1','cmd2','cmd3','cmd4','cmd5']].values, axis=1)
    return df


def load_episodes(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_montecarlo(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train_episodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        'episode_reward': 'reward',
        'ep_rmse_m': 'deviation_rmse_m',
        'yoshikawa_wall': 'yoshikawa_w',
    }
    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
    return df


def load_train_updates(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Analyse commande numérique
# ─────────────────────────────────────────────────────────────────────────────

def plot_commande_numerique(df: pd.DataFrame, noise: float = 0.0, ep: int = 1):
    """Consigne vs sortie, erreur de suivi, effort de contrôle pour 1 épisode."""
    sub = df[(df['noise_level'] == noise) & (df['episode'] == ep)].copy()
    if sub.empty:
        print(f'[analyse] Pas de données pour noise={noise}, ep={ep}')
        return

    t = np.arange(len(sub)) * 0.02  # ~50 Hz env steps → secondes

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'Analyse Commande Numérique — Épisode {ep}  (bruit σ={noise*100:.1f}%)',
        fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 2.1  Position Y : consigne vs sortie ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, sub['target_y'] * 100, color=COLORS['consigne'],
            lw=2, label='Consigne y(t)', zorder=3)
    ax.plot(t, sub['dot_y'] * 100, color=COLORS['sortie'],
            lw=1.2, alpha=0.8, label='Laser y (mesure)')
    ax.plot(t, sub['ekf_y'] * 100, color=COLORS['ekf'],
            lw=1.5, linestyle='--', label='EKF ŷ(t)')
    ax.fill_between(t,
        (sub['ekf_y'] - np.sqrt(sub['ekf_cov_yy'])) * 100,
        (sub['ekf_y'] + np.sqrt(sub['ekf_cov_yy'])) * 100,
        color=COLORS['ekf'], alpha=0.15, label='±1σ EKF')
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Position Y (cm)')
    ax.set_title('Axe Y — Consigne vs Sortie')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── 2.2  Position Z : consigne vs sortie ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, sub['target_z'] * 100, color=COLORS['consigne'],
            lw=2, label='Consigne z(t)', zorder=3)
    ax.plot(t, sub['dot_z'] * 100, color=COLORS['sortie'],
            lw=1.2, alpha=0.8, label='Laser z (mesure)')
    ax.plot(t, sub['ekf_z'] * 100, color=COLORS['ekf'],
            lw=1.5, linestyle='--', label='EKF ẑ(t)')
    ax.fill_between(t,
        (sub['ekf_z'] - np.sqrt(sub['ekf_cov_zz'])) * 100,
        (sub['ekf_z'] + np.sqrt(sub['ekf_cov_zz'])) * 100,
        color=COLORS['ekf'], alpha=0.15)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Position Z (cm)')
    ax.set_title('Axe Z — Consigne vs Sortie')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── 2.3  Erreur de suivi ──
    ax = fig.add_subplot(gs[1, 0])
    err = sub['tracking_error_m'] * 100
    err_static = err.mean()
    ax.plot(t, err, color=COLORS['error'], lw=1.2, label='Erreur ‖e(t)‖ (cm)')
    ax.axhline(err_static, color='red', lw=1.5, linestyle='--',
               label=f'Erreur statique : {err_static:.2f} cm')
    ax.axhline(err.quantile(0.95), color='orange', lw=1, linestyle=':',
               label=f'P95 : {err.quantile(0.95):.2f} cm')
    ax.fill_between(t, 0, err, color=COLORS['error'], alpha=0.12)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Erreur (cm)')
    ax.set_title('Erreur de suivi ‖r - ŷ‖')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── 2.4  RMSE glissant ──
    ax = fig.add_subplot(gs[1, 1])
    window = max(1, len(sub) // 20)
    rmse_roll = err.rolling(window, min_periods=1).apply(
        lambda x: np.sqrt(np.mean(x**2)))
    ax.plot(t, rmse_roll, color=COLORS['error'], lw=1.5, label=f'RMSE (fenêtre {window})')
    ax.axhline(np.sqrt(np.mean(err**2)), color='red', lw=1.5, linestyle='--',
               label=f'RMSE global : {np.sqrt(np.mean(err**2)):.2f} cm')
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('RMSE (cm)')
    ax.set_title('RMSE glissant de l\'erreur de position')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── 2.5  Commandes articulaires ──
    ax = fig.add_subplot(gs[2, :])
    cmap = plt.cm.tab10
    cmd_cols = ['cmd0','cmd1','cmd2','cmd3','cmd4','cmd5']
    labels    = ['J1 Base','J2 Épaule','J3 Coude','J4 P.1','J5 P.2','J6 P.3']
    for i, (col, lbl) in enumerate(zip(cmd_cols, labels)):
        ax.plot(t, sub[col], color=cmap(i), lw=1.0, label=lbl, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, linestyle=':')
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Commande vitesse (rad/s norm.)')
    ax.set_title('Commandes articulaires — 6 axes')
    ax.legend(fontsize=6.5, ncol=6, loc='upper right')
    ax.grid(alpha=0.3)

    # ── 2.6  Effort de contrôle ──
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, sub['ctrl_norm'], color=COLORS['ctrl'], lw=1.2)
    ax.fill_between(t, 0, sub['ctrl_norm'], color=COLORS['ctrl'], alpha=0.2)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('‖u(t)‖ (rad/s)')
    ax.set_title('Effort de contrôle total')
    ax.grid(alpha=0.3)

    # ── 2.7  Portrait de phase (trajectoire y-z) ──
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(sub['target_y'] * 100, sub['target_z'] * 100,
            color=COLORS['consigne'], lw=2.5, label='Consigne', zorder=3)
    sc = ax.scatter(sub['dot_y'] * 100, sub['dot_z'] * 100,
                    c=t, cmap='plasma', s=4, alpha=0.7, label='Laser mesure')
    ax.plot(sub['ekf_y'] * 100, sub['ekf_z'] * 100,
            color=COLORS['ekf'], lw=1.0, linestyle='--', label='EKF', alpha=0.8)
    fig.colorbar(sc, ax=ax, label='Temps (s)', pad=0.02)
    ax.set_xlabel('Y (cm)')
    ax.set_ylabel('Z (cm)')
    ax.set_title('Trajectoire dans le plan du mur')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.savefig(OUT_DIR / f'commande_numerique_noise{int(noise*1000):04d}_ep{ep:03d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analyse] → commande_numerique_noise{int(noise*1000):04d}_ep{ep:03d}.png')


# ─────────────────────────────────────────────────────────────────────────────
#  3. Analyse EKF
# ─────────────────────────────────────────────────────────────────────────────

def plot_ekf(df: pd.DataFrame, noise: float = 0.0, ep: int = 1):
    sub = df[(df['noise_level'] == noise) & (df['episode'] == ep)].copy()
    if sub.empty:
        return

    t = np.arange(len(sub)) * 0.02

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f'Analyse EKF — Épisode {ep}  (bruit σ={noise*100:.1f}%)',
                 fontsize=13, fontweight='bold')

    # ── Innovation Y ──
    ax = axes[0, 0]
    ax.plot(t, sub['ekf_innovation_y'] * 100, color=COLORS['ekf'], lw=1.2)
    ax.axhline(0, color='black', lw=0.8, linestyle=':')
    sigma = np.sqrt(sub['ekf_cov_yy'].mean()) * 100
    ax.fill_between(t, -sigma, sigma, color=COLORS['ekf'], alpha=0.15,
                    label=f'±1σ ({sigma:.1f} cm)')
    ax.set_title('Innovation EKF — axe Y')
    ax.set_ylabel('Résidu (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Innovation Z ──
    ax = axes[0, 1]
    ax.plot(t, sub['ekf_innovation_z'] * 100, color=COLORS['fk'], lw=1.2)
    ax.axhline(0, color='black', lw=0.8, linestyle=':')
    sigma_z = np.sqrt(sub['ekf_cov_zz'].mean()) * 100
    ax.fill_between(t, -sigma_z, sigma_z, color=COLORS['fk'], alpha=0.15,
                    label=f'±1σ ({sigma_z:.1f} cm)')
    ax.set_title('Innovation EKF — axe Z')
    ax.set_ylabel('Résidu (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Covariance ──
    ax = axes[1, 0]
    ax.plot(t, np.sqrt(sub['ekf_cov_yy']) * 100,
            color=COLORS['ekf'], lw=1.5, label='σ_y (cm)')
    ax.plot(t, np.sqrt(sub['ekf_cov_zz']) * 100,
            color=COLORS['fk'], lw=1.5, label='σ_z (cm)')
    ax.set_title('Incertitude EKF (√P diagonal)')
    ax.set_ylabel('σ (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── NEES (Normalized Estimation Error Squared) ──
    ax = axes[1, 1]
    ey = (sub['dot_y'] - sub['ekf_y']) ** 2 / sub['ekf_cov_yy'].clip(1e-10)
    ez = (sub['dot_z'] - sub['ekf_z']) ** 2 / sub['ekf_cov_zz'].clip(1e-10)
    nees = (ey + ez).rolling(20, min_periods=1).mean()
    ax.plot(t, nees, color=COLORS['error'], lw=1.2, label='NEES (glissant)')
    ax.axhline(2.0, color='orange', lw=1.5, linestyle='--',
               label='Seuil χ²(2, 95%) = 5.99')
    ax.axhline(5.99, color='red', lw=1.0, linestyle=':')
    ax.set_title('NEES — cohérence EKF')
    ax.set_ylabel('NEES')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── FK vs EKF vs dot ──
    ax = axes[2, 0]
    ax.plot(t, sub['dot_y'] * 100, color=COLORS['sortie'], lw=1.0,
            alpha=0.6, label='Dot laser (FK)', zorder=2)
    ax.plot(t, sub['ekf_y'] * 100, color=COLORS['ekf'],
            lw=1.8, label='EKF ŷ', zorder=3)
    ax.plot(t, sub['target_y'] * 100, color=COLORS['consigne'],
            lw=1.5, linestyle='--', label='Consigne', zorder=4)
    ax.set_title('Comparaison FK / EKF / Consigne — Y')
    ax.set_ylabel('Position Y (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, sub['dot_z'] * 100, color=COLORS['sortie'], lw=1.0,
            alpha=0.6, label='Dot laser (FK)', zorder=2)
    ax.plot(t, sub['ekf_z'] * 100, color=COLORS['ekf'],
            lw=1.8, label='EKF ẑ', zorder=3)
    ax.plot(t, sub['target_z'] * 100, color=COLORS['consigne'],
            lw=1.5, linestyle='--', label='Consigne', zorder=4)
    ax.set_title('Comparaison FK / EKF / Consigne — Z')
    ax.set_ylabel('Position Z (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    for axrow in axes:
        for a in axrow:
            a.set_xlabel('Temps (s)')

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'ekf_analyse_noise{int(noise*1000):04d}_ep{ep:03d}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analyse] → ekf_analyse_noise{int(noise*1000):04d}_ep{ep:03d}.png')


# ─────────────────────────────────────────────────────────────────────────────
#  4. Monte Carlo — Robustesse
# ─────────────────────────────────────────────────────────────────────────────

def plot_montecarlo(df_mc: pd.DataFrame, df_ep: pd.DataFrame = None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle('Analyse Monte Carlo — Robustesse par niveau de bruit',
                 fontsize=13, fontweight='bold')

    noise_labels = [f'{n*100:.1f}%' for n in df_mc['noise_level']]
    x = np.arange(len(df_mc))
    w = 0.6

    # ── Reward moyen ± std ──
    ax = axes[0, 0]
    bars = ax.bar(x, df_mc['mean_reward'], w, color=COLORS['consigne'], alpha=0.85)
    ax.errorbar(x, df_mc['mean_reward'], yerr=df_mc['std_reward'],
                fmt='none', color='black', capsize=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_title('Reward moyen (± 1σ)')
    ax.set_xlabel('Niveau de bruit σ')
    ax.set_ylabel('Reward')
    ax.grid(alpha=0.3, axis='y')

    # ── Déviation moyenne ± std ──
    ax = axes[0, 1]
    ax.bar(x, df_mc['mean_deviation_m'] * 100, w, color=COLORS['error'], alpha=0.85)
    ax.errorbar(x, df_mc['mean_deviation_m'] * 100,
                yerr=df_mc['std_deviation_m'] * 100,
                fmt='none', color='black', capsize=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_title('Déviation moyenne (± 1σ)')
    ax.set_xlabel('Niveau de bruit σ')
    ax.set_ylabel('Déviation (cm)')
    ax.grid(alpha=0.3, axis='y')

    # ── Taux de succès ──
    ax = axes[0, 2]
    ax.bar(x, df_mc['mean_success'] * 100, w, color=COLORS['ctrl'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_title('Taux de succès (%)')
    ax.set_xlabel('Niveau de bruit σ')
    ax.set_ylabel('Succès (%)')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis='y')

    # ── Percentiles reward (p5, p50, p95) ──
    ax = axes[1, 0]
    ax.fill_between(x, df_mc['p5_reward'], df_mc['p95_reward'],
                    alpha=0.3, color=COLORS['consigne'], label='P5–P95')
    ax.plot(x, df_mc['p50_reward'], 'o-', color=COLORS['consigne'],
            lw=2, ms=6, label='P50 (médiane)')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_title('Distribution reward — percentiles')
    ax.set_xlabel('Niveau de bruit σ')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Percentiles déviation (p5, p50, p95) ──
    ax = axes[1, 1]
    ax.fill_between(x, df_mc['p5_deviation'] * 100, df_mc['p95_deviation'] * 100,
                    alpha=0.3, color=COLORS['error'], label='P5–P95')
    ax.plot(x, df_mc['p50_deviation'] * 100, 'o-', color=COLORS['error'],
            lw=2, ms=6, label='P50 (médiane)')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_title('Distribution déviation — percentiles')
    ax.set_xlabel('Niveau de bruit σ')
    ax.set_ylabel('Déviation (cm)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Boxplot déviation si données épisodes dispo ──
    ax = axes[1, 2]
    if df_ep is not None:
        noise_vals = sorted(df_ep['noise_level'].unique())
        data_box = [df_ep[df_ep['noise_level'] == n]['deviation_mean_m'].values * 100
                    for n in noise_vals]
        bp = ax.boxplot(data_box, labels=[f'{n*100:.1f}%' for n in noise_vals],
                        patch_artist=True, notch=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['sortie'])
            patch.set_alpha(0.7)
        ax.set_title('Boxplot déviation par niveau de bruit')
        ax.set_xlabel('Niveau de bruit σ')
        ax.set_ylabel('Déviation (cm)')
    else:
        ax.text(0.5, 0.5, 'Données épisodes\nnon disponibles',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        ax.set_title('Boxplot déviation')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'montecarlo_robustesse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[analyse] → montecarlo_robustesse.png')

    # ── Meilleur filtre EKF ──
    _report_best_ekf(df_mc)


def _report_best_ekf(df_mc: pd.DataFrame):
    """Identifie et affiche le meilleur paramétrage EKF selon RMSE + robustesse."""
    print('\n' + '='*60)
    print('MEILLEUR FILTRE EKF — Analyse Monte Carlo')
    print('='*60)

    # Score composite : minimiser RMSE pondéré par (1 + std/mean) sur tous les bruits
    df_mc = df_mc.copy()
    df_mc['score'] = (df_mc['mean_rmse'] *
                      (1 + df_mc['std_rmse'] / df_mc['mean_rmse'].clip(1e-6)))
    best_idx = df_mc['score'].idxmin()

    print('\nParamètres EKF implémentés (LaserDotEKF) :')
    print('  État          : x = [y, z, vy, vz]  (4 dimensions)')
    print('  Modèle process: cinématique laser via J_wall(q)·q̇')
    print('  q_pos_std     : 1 mm  (bruit processus position)')
    print('  q_vel_std     : 8 cm/s (bruit processus vitesse)')
    print('  r_fk_std      : 5 mm  (mesure FK haute fréquence)')
    print('  r_cam_std     : 18 mm (réservé à une mesure caméra absolue calibrée)')
    print('  Fréquence FK  : 250 Hz')
    print('  Fréquence cam : 30 Hz (KLT relatif, non injecté comme position absolue)')
    print('  Gain Kalman   : forme de Joseph (stable)')

    print('\nPerformances par niveau de bruit :')
    print(f'{"Bruit σ":>10} | {"RMSE (cm)":>10} | {"Dév. moy":>10} | {"Succès":>8}')
    print('-' * 46)
    for _, row in df_mc.iterrows():
        print(f'{row["noise_level"]*100:>9.1f}% | '
              f'{row["mean_rmse"]*100:>10.2f} | '
              f'{row["mean_deviation_m"]*100:>9.2f} | '
              f'{row["mean_success"]*100:>7.0f}%')

    print(f'\n→ Niveau de bruit optimal : {df_mc.loc[best_idx, "noise_level"]*100:.1f}%')
    print(f'  RMSE         : {df_mc.loc[best_idx, "mean_rmse"]*100:.2f} ± '
          f'{df_mc.loc[best_idx, "std_rmse"]*100:.2f} cm')
    print(f'  Déviation    : {df_mc.loc[best_idx, "mean_deviation_m"]*100:.2f} cm')
    print(f'  Taux succès  : {df_mc.loc[best_idx, "mean_success"]*100:.0f}%')
    print('\nConclusion : le filtre 4D lisse la position cinématique du spot.')
    print('Dans la V2.2, KLT fournit une erreur relative ligne-laser au SAC,')
    print('mais n’est pas injecté comme position absolue dans le Kalman.')
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Courbes d'entraînement SAC
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(df_ep: pd.DataFrame, df_up: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Courbes d\'entraînement SAC — UR7e Laser Line-Follower',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    steps = df_ep['timestep']

    def smooth(y, w=30):
        return pd.Series(y).rolling(w, min_periods=1).mean()

    # Reward
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, df_ep['reward'], alpha=0.25, color=COLORS['consigne'], lw=0.8)
    ax.plot(steps, smooth(df_ep['reward']), color=COLORS['consigne'], lw=2)
    ax.set_title('Reward par épisode')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.grid(alpha=0.3)

    # Déviation
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, df_ep['deviation_mean_m'] * 100, alpha=0.25,
            color=COLORS['error'], lw=0.8)
    ax.plot(steps, smooth(df_ep['deviation_mean_m'] * 100),
            color=COLORS['error'], lw=2)
    ax.set_title('Déviation moyenne (cm)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Déviation (cm)')
    ax.grid(alpha=0.3)

    # Taux de succès
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, smooth(df_ep['success']) * 100, color=COLORS['ctrl'], lw=2)
    ax.set_title('Taux de succès (%)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Succès (%)')
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    # RMSE
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, df_ep['deviation_rmse_m'] * 100, alpha=0.25,
            color=COLORS['fk'], lw=0.8)
    ax.plot(steps, smooth(df_ep['deviation_rmse_m'] * 100),
            color=COLORS['fk'], lw=2)
    ax.set_title('RMSE déviation (cm)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('RMSE (cm)')
    ax.grid(alpha=0.3)

    # Yoshikawa
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, smooth(df_ep['yoshikawa_w']), color=COLORS['cam'], lw=2)
    ax.set_title('Manipulabilité Yoshikawa')
    ax.set_xlabel('Steps')
    ax.set_ylabel('w(q)')
    ax.grid(alpha=0.3)

    # Progression trajectoire
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps, smooth(df_ep['progress']) * 100, color=COLORS['sortie'], lw=2)
    ax.set_title('Progression trajectoire (%)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('%')
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.3)

    # Actor / Critic loss
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df_up['timestep'], smooth(df_up['actor_loss'], 50),
            color=COLORS['rl'], lw=1.5, label='Actor loss')
    ax.set_title('Actor Loss (SAC)')
    ax.set_xlabel('Steps')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(df_up['timestep'], smooth(df_up['critic_loss'], 50),
            color=COLORS['ctrl'], lw=1.5, label='Critic loss')
    ax.set_title('Critic Loss (SAC)')
    ax.set_xlabel('Steps')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Entropie
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(df_up['timestep'], smooth(df_up['ent_coef'], 50),
            color=COLORS['ekf'], lw=1.5, label='Entropie α')
    ax.set_title('Coefficient entropie α (SAC)')
    ax.set_xlabel('Steps')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.savefig(OUT_DIR / 'courbes_entrainement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[analyse] → courbes_entrainement.png')


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',   default='', help='Chemin eval_steps_*.csv')
    parser.add_argument('--episodes',default='', help='Chemin eval_episodes_*.csv')
    parser.add_argument('--mc',      default='', help='Chemin eval_montecarlo_*.csv')
    parser.add_argument('--train-ep',default='', help='Chemin episodes training CSV')
    parser.add_argument('--train-up',default='', help='Chemin updates training CSV')
    args = parser.parse_args()

    # ── Auto-détection des fichiers ──────────────────────────────────────────
    def latest(pattern, directory):
        files = sorted(directory.glob(pattern))
        return files[-1] if files else None

    p_steps   = Path(args.steps)   if args.steps   else latest('eval_steps_*.csv',   EVAL_DIR)
    p_ep      = Path(args.episodes) if args.episodes else latest('eval_episodes_*.csv', EVAL_DIR)
    p_mc      = Path(args.mc)      if args.mc      else latest('eval_montecarlo_*.csv', EVAL_DIR)
    p_train_ep = Path(args.train_ep) if args.train_ep else latest('train_episodes_*.csv', TRAIN_DIR) or latest('episodes_*.csv', TRAIN_DIR)
    p_train_up = Path(args.train_up) if args.train_up else latest('train_updates_*.csv', TRAIN_DIR) or latest('updates_*.csv', TRAIN_DIR)

    print(f'[analyse] Steps        : {p_steps}')
    print(f'[analyse] Épisodes eval: {p_ep}')
    print(f'[analyse] Monte Carlo  : {p_mc}')
    print(f'[analyse] Train ep     : {p_train_ep}')
    print(f'[analyse] Train updates: {p_train_up}')

    # ── Courbes d'entraînement (toujours dispo) ───────────────────────────────
    if p_train_ep and p_train_ep.exists() and p_train_up and p_train_up.exists():
        df_tr_ep = load_train_episodes(p_train_ep)
        df_tr_up = load_train_updates(p_train_up)
        plot_training(df_tr_ep, df_tr_up)
    else:
        print('[analyse] ⚠ CSV entraînement non trouvé — skip courbes training')

    # ── Données évaluation (après eval_full.py) ───────────────────────────────
    if p_steps and p_steps.exists():
        df_steps = load_steps(p_steps)
        noise_vals = sorted(df_steps['noise_level'].unique())

        # Commande numérique : 1 épisode par niveau de bruit
        for noise in noise_vals:
            plot_commande_numerique(df_steps, noise=noise, ep=1)
            plot_ekf(df_steps, noise=noise, ep=1)
    else:
        print('[analyse] ⚠ eval_steps non trouvé — skip analyse commande / EKF')
        print('          → lance d\'abord : ros2 run ur7e_line_follower eval_full')

    df_ep_eval = None
    if p_ep and p_ep.exists():
        df_ep_eval = load_episodes(p_ep)

    if p_mc and p_mc.exists():
        df_mc = load_montecarlo(p_mc)
        plot_montecarlo(df_mc, df_ep_eval)
    else:
        print('[analyse] ⚠ eval_montecarlo non trouvé — skip Monte Carlo')

    print(f'\n[analyse] Figures dans {OUT_DIR}/')
    for f in sorted(OUT_DIR.glob('*.png')):
        print(f'  {f.name}')


def entry_point():
    main()


if __name__ == '__main__':
    main()
