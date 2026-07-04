"""Générateur de courbes PNG/HTML depuis les CSV train/eval/tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'
OUT_DIR = DATA_DIR / 'plots'

def _default_metrics_dir() -> Path:
    latest = DATA_DIR / 'latest_run.txt'
    if latest.exists():
        run_dir = Path(latest.read_text().strip()).expanduser()
        if (run_dir / 'metrics').is_dir():
            return run_dir / 'metrics'
    return DATA_DIR / 'metrics'


def _latest(pattern: str, folder: Path):
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None


def _save(fig, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_line(df, x, ys, title, ylabel, name, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in ys:
        if y in df.columns:
            ax.plot(df[x], df[y], label=y)
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.legend()
    return _save(fig, name, out_dir)


def _plot_scatter(df, x, y, c, title, name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df[x], df[y], c=df[c] if c in df.columns else None, s=16, alpha=0.75)
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True, alpha=0.3)
    if c in df.columns: fig.colorbar(sc, ax=ax, label=c)
    return _save(fig, name, out_dir)


def plot_training(metrics_dir: Path, out_dir: Path):
    paths = []
    ep = _latest('train_episodes_*.csv', metrics_dir)
    win = _latest('train_windows_*.csv', metrics_dir)
    up = _latest('train_updates_*.csv', metrics_dir)
    if win:
        df = pd.read_csv(win)
        paths.append(_plot_line(df, 'timestep', ['reward_mean'], 'RL — reward moyen par fenêtre', 'reward', 'rl_reward.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['success_rate', 'progress_mean'], 'RL — succès et progression', 'ratio', 'rl_success_progress.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['rmse_mean_m', 'rmse_p50_m', 'rmse_p95_m'], 'RL — RMSE tracking', 'm', 'rl_rmse.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['offwall_ratio_mean'], 'RL — ratio hors mur', 'ratio', 'rl_offwall.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['cmd_out_norm_mean'], 'Commande — norme moyenne sortie', 'rad/s', 'rl_command_norm.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['ekf_sigma_mean_m'], 'Kalman — sigma moyen pendant training', 'm', 'rl_ekf_sigma.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['yoshikawa_wall_mean', 'cond_wall_median'], 'Singularités — manipulabilité/conditionnement', 'valeur', 'rl_singularity.png', out_dir))
    if up:
        df = pd.read_csv(up)
        paths.append(_plot_line(df, 'timestep', ['actor_loss', 'critic_loss'], 'SAC — pertes actor/critic', 'loss', 'sac_losses.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['ent_coef', 'ent_coef_loss'], 'SAC — entropie', 'valeur', 'sac_entropy.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['learning_rate'], 'SAC — learning rate', 'lr', 'sac_learning_rate.png', out_dir))
        paths.append(_plot_line(df, 'timestep', ['buffer_size', 'n_updates'], 'SAC — replay buffer / updates', 'count', 'sac_buffer_updates.png', out_dir))
    if ep:
        df = pd.read_csv(ep)
        if 'ep_rmse_m' in df.columns and 'episode_reward' in df.columns:
            paths.append(_plot_scatter(df, 'ep_rmse_m', 'episode_reward', 'progress', 'RL — reward vs RMSE', 'rl_reward_vs_rmse.png', out_dir))
    return paths


def plot_eval(eval_dir: Path, out_dir: Path):
    paths = []
    ep = _latest('eval_episodes_*.csv', eval_dir)
    steps = _latest('eval_steps_*.csv', eval_dir)
    mc = _latest('eval_montecarlo_*.csv', eval_dir)
    if mc:
        df = pd.read_csv(mc)
        paths.append(_plot_line(df, 'noise_level_m', ['success_rate', 'progress_mean'], 'Monte-Carlo — succès/progression vs bruit', 'ratio', 'mc_success_progress.png', out_dir))
        paths.append(_plot_line(df, 'noise_level_m', ['rmse_mean_m', 'rmse_p50_m', 'rmse_p95_m'], 'Monte-Carlo — RMSE vs bruit', 'm', 'mc_rmse_noise.png', out_dir))
        paths.append(_plot_line(df, 'noise_level_m', ['reward_mean', 'reward_p5', 'reward_p95'], 'Monte-Carlo — reward vs bruit', 'reward', 'mc_reward_noise.png', out_dir))
    if ep:
        df = pd.read_csv(ep)
        if 'noise_level_m' in df.columns:
            fig, ax = plt.subplots(figsize=(9,5))
            groups = [g['rmse_m'].dropna().to_numpy() for _, g in df.groupby('noise_level_m')]
            labels = [f'{k*1000:.0f}mm' for k in sorted(df['noise_level_m'].unique())]
            if groups:
                ax.boxplot(groups, labels=labels)
            ax.set_title('Monte-Carlo — distribution RMSE par bruit'); ax.set_xlabel('bruit'); ax.set_ylabel('RMSE (m)'); ax.grid(True, alpha=0.3)
            paths.append(_save(fig, 'mc_rmse_boxplot.png', out_dir))
    if steps:
        df = pd.read_csv(steps)
        # Premier épisode du premier niveau de bruit pour les courbes temporelles.
        n0 = df['noise_level_m'].iloc[0]; e0 = df['episode'].iloc[0]
        d0 = df[(df['noise_level_m'] == n0) & (df['episode'] == e0)].copy()
        if not d0.empty:
            paths.append(_plot_line(d0, 'step', ['tracking_error_m'], 'Test — erreur de suivi step-by-step', 'm', 'eval_tracking_error.png', out_dir))
            paths.append(_plot_line(d0, 'step', ['ekf_sigma_y','ekf_sigma_z'], 'Kalman — incertitude', 'm', 'eval_ekf_sigma.png', out_dir))
            paths.append(_plot_line(d0, 'step', ['ekf_innov_y','ekf_innov_z'], 'Kalman — innovations', 'm', 'eval_ekf_innovation.png', out_dir))
            paths.append(_plot_line(d0, 'step', ['cmd_raw0','cmd_lqr0','cmd_out0','cmd_null0'], 'LQR — commande joint 0', 'rad/s', 'eval_lqr_joint0.png', out_dir))
            norms = pd.DataFrame({'step': d0['step']})
            for prefix in ['cmd_raw','cmd_lqr','cmd_null','cmd_out']:
                cols = [f'{prefix}{i}' for i in range(6) if f'{prefix}{i}' in d0.columns]
                norms[prefix + '_norm'] = np.linalg.norm(d0[cols].to_numpy(), axis=1) if cols else np.nan
            paths.append(_plot_line(norms, 'step', ['cmd_raw_norm','cmd_lqr_norm','cmd_null_norm','cmd_out_norm'], 'LQR — normes de commande', 'rad/s', 'eval_lqr_norms.png', out_dir))
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(d0['dot_y'], d0['dot_z'], label='laser dot')
            ax.plot(d0['closest_y'], d0['closest_z'], label='closest line point')
            ax.scatter(d0['target_y'], d0['target_z'], s=8, alpha=0.35, label='target waypoint')
            ax.set_title('Trajectoire 2D sur le mur'); ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)'); ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend()
            paths.append(_save(fig, 'eval_wall_trajectory_2d.png', out_dir))
            try:
                import plotly.graph_objects as go
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter3d(x=d0['step'], y=d0['dot_y'], z=d0['dot_z'], mode='lines', name='laser dot'))
                fig3.add_trace(go.Scatter3d(x=d0['step'], y=d0['closest_y'], z=d0['closest_z'], mode='lines', name='closest line'))
                fig3.update_layout(title='Trajectoire 3D interactive : step-y-z', scene=dict(xaxis_title='step', yaxis_title='y', zaxis_title='z'))
                html = out_dir / 'eval_wall_trajectory_3d.html'; fig3.write_html(str(html)); paths.append(html)
            except Exception:
                pass
    return paths


def plot_component_tests(test_dir: Path, out_dir: Path):
    paths = []
    kin = _latest('kinematics_*.csv', test_dir)
    kal = _latest('kalman_*.csv', test_dir)
    lqr = _latest('lqr_*.csv', test_dir)
    mc = _latest('montecarlo_lines_*.csv', test_dir)
    if kin:
        df = pd.read_csv(kin)
        paths.append(_plot_scatter(df, 'cond_wall', 'wall_vel_err', 'on_wall', 'MGD/MGI — erreur vitesse mur vs conditionnement', 'diag_mgi_wall_error.png', out_dir))
        paths.append(_plot_scatter(df, 'tcp_y', 'tcp_z', 'cond_tcp', 'MGD — workspace TCP coloré conditionnement', 'diag_mgd_workspace.png', out_dir))
    if kal:
        df = pd.read_csv(kal)
        paths.append(_plot_line(df, 'step', ['true_y','ekf_y','meas_fk_y'], 'Kalman test — y vrai/FK/EKF', 'm', 'diag_kalman_y.png', out_dir))
        paths.append(_plot_line(df, 'step', ['err_m','sigma_y','sigma_z'], 'Kalman test — erreur et sigma', 'm', 'diag_kalman_error_sigma.png', out_dir))
        paths.append(_plot_line(df, 'step', ['nis','nees'], 'Kalman test — NIS/NEES', 'stat', 'diag_kalman_nis_nees.png', out_dir))
    if lqr:
        df = pd.read_csv(lqr)
        paths.append(_plot_scatter(df, 'w_wall', 'gain_max', 'cond_wall', 'LQR — gain max vs manipulabilité', 'diag_lqr_gain.png', out_dir))
        paths.append(_plot_scatter(df, 'raw_norm', 'lqr_norm', 'w_wall', 'LQR — norme avant/après', 'diag_lqr_norm.png', out_dir))
        paths.append(_plot_scatter(df, 'wall_speed_raw', 'wall_speed_null', 'w_wall', 'Noyau — vitesse spot résiduelle', 'diag_nullspace_residual.png', out_dir))
    if mc:
        df = pd.read_csv(mc)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df['length_m'], bins=30)
        ax.set_title('Monte-Carlo lignes — distribution longueurs'); ax.set_xlabel('longueur (m)'); ax.set_ylabel('count'); ax.grid(True, alpha=0.3)
        paths.append(_save(fig, 'diag_lines_length_hist.png', out_dir))
        paths.append(_plot_scatter(df, 'length_m', 'max_spacing_m', 'in_bounds', 'Monte-Carlo lignes — longueur vs espacement max', 'diag_lines_spacing.png', out_dir))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-dir', default=str(_default_metrics_dir()))
    parser.add_argument('--eval-dir', default=str(DATA_DIR / 'eval'))
    parser.add_argument('--test-dir', default=str(DATA_DIR / 'tests'))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    paths += plot_training(Path(args.metrics_dir), out)
    paths += plot_eval(Path(args.eval_dir), out)
    paths += plot_component_tests(Path(args.test_dir), out)
    print('[plot_results] figures générées :')
    for p in paths: print(f'  {p}')


if __name__ == '__main__':
    main()
