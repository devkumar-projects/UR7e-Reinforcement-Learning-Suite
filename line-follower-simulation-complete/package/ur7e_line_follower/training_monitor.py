"""
training_monitor.py — Dashboard terminal temps-réel pendant l'entraînement SAC.

Lit les CSV générés par MetricsCallback toutes les 5 secondes et affiche
un tableau de bord synthétique avec rich.

Usage :
    ros2 run ur7e_line_follower training_monitor
    # ou directement :
    python3 -m ur7e_line_follower.training_monitor
    python3 -m ur7e_line_follower.training_monitor --refresh 10
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from collections import deque

import numpy as np

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'
LEGACY_METRICS_DIR = DATA_DIR / 'metrics'


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


def _requested_total(metrics_dir: Path) -> int:
    config = metrics_dir.parent / 'config.json'
    if config.exists():
        try:
            return max(1, int(json.loads(config.read_text()).get('total_timesteps_requested', 1)))
        except Exception:
            pass
    return 1

console = Console()


# ── Lecture CSV ────────────────────────────────────────────────────────────────

def _latest(pattern: str, folder: Path) -> Path | None:
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None


def _read_tail(path: Path, n: int = 200) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-n:]


def _float(row: dict, key: str, default: float = float('nan')) -> float:
    try:
        v = row.get(key, '')
        return float(v) if v not in ('', 'nan', 'None', None) else default
    except (ValueError, TypeError):
        return default


def _int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (ValueError, TypeError):
        return default


# ── Calculs statistiques ───────────────────────────────────────────────────────

def _smooth(vals: list[float], w: int = 10) -> float:
    arr = [v for v in vals if not (v != v)]  # drop NaN
    if not arr:
        return float('nan')
    return float(np.mean(arr[-w:]))


def _pct_bar(ratio: float, width: int = 20, color_ok: str = 'green',
             color_warn: str = 'yellow', color_bad: str = 'red') -> Text:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    bar = '█' * filled + '░' * (width - filled)
    color = color_ok if ratio > 0.6 else (color_warn if ratio > 0.3 else color_bad)
    return Text(f'[{bar}] {ratio*100:5.1f}%', style=color)


# ── Panneaux individuels ───────────────────────────────────────────────────────

def _panel_rl(ep_rows: list[dict], win_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Métrique', style='bold cyan', width=28)
    t.add_column('Valeur', width=22)
    t.add_column('Tendance (50 ep)', width=24)

    if not ep_rows:
        return Panel('[dim]En attente des données…[/dim]', title='[bold yellow]RL — SAC[/bold yellow]')

    step = _int(ep_rows[-1], 'timestep')
    ep   = _int(ep_rows[-1], 'episode')

    rewards  = [_float(r, 'episode_reward') for r in ep_rows]
    progresses = [_float(r, 'progress')     for r in ep_rows]
    successes  = [_float(r, 'success')      for r in ep_rows]
    rmses      = [_float(r, 'ep_rmse_m') * 100 for r in ep_rows]
    devs       = [_float(r, 'deviation_mean_m') * 100 for r in ep_rows]
    lengths    = [_float(r, 'episode_length') for r in ep_rows]

    rew50   = _smooth(rewards, 50)
    prog50  = _smooth(progresses, 50)
    succ50  = _smooth(successes, 50)
    rmse50  = _smooth(rmses, 50)
    dev50   = _smooth(devs, 50)
    len50   = _smooth(lengths, 50)

    cur_lev = _int(ep_rows[-1], 'curriculum_level', 0)
    stag    = _int(ep_rows[-1], 'stagnation_steps', 0)

    t.add_row('Timestep / Épisode',
              f'[bold]{step:,}[/bold] / [bold]{ep}[/bold]', '')
    t.add_row('Récompense moy. (50 ep)', f'{rew50:+.2f}',
              _pct_bar((rew50 + 5) / 25.0))
    t.add_row('Taux de succès (50 ep)',  f'{succ50*100:.1f}%',
              _pct_bar(succ50))
    t.add_row('Progression moy. (50 ep)', f'{prog50*100:.1f}%',
              _pct_bar(prog50))
    t.add_row('RMSE laser↔cible (50 ep)', f'{rmse50:.1f} cm',
              _pct_bar(1.0 - rmse50/10.0, color_ok='green', color_bad='red'))
    t.add_row('Écart moyen (50 ep)',      f'{dev50:.1f} cm', '')
    t.add_row('Durée moy. épisode',       f'{len50:.0f} steps', '')
    t.add_row('Curriculum level',         f'{cur_lev}/2', '')
    t.add_row('Stagnation steps',         f'{stag}', '')

    return Panel(t, title='[bold yellow]■ RL — SAC[/bold yellow]',
                 border_style='yellow')


def _panel_sac_internals(up_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Paramètre', style='bold cyan', width=22)
    t.add_column('Valeur courante', width=18)
    t.add_column('Tendance (20)', width=24)

    if not up_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold yellow]SAC — Internals[/bold yellow]')

    actor_losses  = [_float(r, 'actor_loss')   for r in up_rows]
    critic_losses = [_float(r, 'critic_loss')  for r in up_rows]
    ent_coefs     = [_float(r, 'ent_coef')     for r in up_rows]
    n_updates_vals= [_float(r, 'n_updates')    for r in up_rows]
    buf_sizes     = [_float(r, 'buffer_size')  for r in up_rows]

    t.add_row('Actor loss',    f'{_smooth(actor_losses, 5):.4f}',
              _pct_bar(max(0, 1 - abs(_smooth(actor_losses, 5)) / 2.0)))
    t.add_row('Critic loss',   f'{_smooth(critic_losses, 5):.4f}',
              _pct_bar(max(0, 1 - _smooth(critic_losses, 5) / 5.0)))
    t.add_row('Entropie α',    f'{_smooth(ent_coefs, 5):.5f}', '')
    t.add_row('Nb updates',    f'{int(_float(up_rows[-1], "n_updates", 0)):,}', '')
    t.add_row('Replay buffer', f'{int(_float(up_rows[-1], "buffer_size", 0)):,}', '')

    return Panel(t, title='[bold yellow]SAC — Internals[/bold yellow]', border_style='yellow')


def _panel_reward_breakdown(ep_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Composante', style='bold cyan', width=28)
    t.add_column('Moy. 20 ep', width=14)
    t.add_column('Contribution', width=22)

    if not ep_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold yellow]Décomposition Reward[/bold yellow]')

    components = {
        'Distance (−dist_ordered)': 'reward_dist_ordered',
        'Waypoint bonus':           'reward_waypoint_bonus',
        'Completion bonus':         'reward_completion_bonus',
        'Progress reward':          'reward_progress_reward',
        'Record bonus':             'reward_record_bonus',
        'Cmd penalty':              'reward_cmd_penalty',
        'Action Δ penalty':         'reward_action_delta_penalty',
        'Stagnation penalty':       'reward_stagnation_penalty',
        'Singularité penalty':      'reward_sing_penalty',
        'Vision penalty':           'reward_vision_penalty',
        'Offwall penalty':          'reward_offwall_penalty',
    }
    total = 0.0
    vals = {}
    for label, key in components.items():
        v = _smooth([_float(r, key, 0.0) for r in ep_rows], 20)
        vals[label] = v
        total += abs(v)

    for label, v in vals.items():
        ratio = abs(v) / max(total, 1e-9)
        color = 'green' if v >= 0 else 'red'
        t.add_row(label, f'[{color}]{v:+.3f}[/{color}]',
                  Text('█' * int(ratio * 20), style=color))

    return Panel(t, title='[bold yellow]Décomposition Reward (20 ep)[/bold yellow]',
                 border_style='yellow')


def _panel_kalman(ep_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Métrique EKF', style='bold magenta', width=26)
    t.add_column('Valeur', width=16)
    t.add_column('Tendance', width=24)

    if not ep_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold magenta]Kalman EKF[/bold magenta]')

    sigmas = [_float(r, 'ekf_sigma_mean_m') * 1000 for r in ep_rows]  # en mm
    nis_vals = [_float(r, 'ekf_nis') for r in ep_rows]

    sigma50 = _smooth(sigmas, 50)
    nis50   = _smooth([v for v in nis_vals if not (v != v)], 20)

    t.add_row('Incertitude σ moy. (50 ep)',
              f'{sigma50:.2f} mm',
              _pct_bar(1.0 - min(sigma50 / 20.0, 1.0),
                       color_ok='green', color_bad='red'))
    t.add_row('NIS moyen (20 ep)',
              f'{nis50:.3f}',
              '[green]cohérent[/green]' if 0.5 < nis50 < 5.0 else '[red]incohérent[/red]')
    t.add_row('Source update', 'FK (250 Hz)', '')
    t.add_row('Modèle pred.', 'Jacobien mur', '')

    return Panel(t, title='[bold magenta]■ Kalman EKF[/bold magenta]',
                 border_style='magenta')


def _panel_klt(ep_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Métrique KLT', style='bold blue', width=26)
    t.add_column('Valeur', width=16)
    t.add_column('Tendance', width=24)

    if not ep_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold blue]KLT Caméra[/bold blue]')

    detects  = [_float(r, 'cam_detected', 0.0)      for r in ep_rows]
    lasers   = [_float(r, 'cam_laser_visible', 0.0) for r in ep_rows]

    det50  = _smooth(detects, 50)
    las50  = _smooth(lasers, 50)

    t.add_row('Ligne détectée (50 ep)',   f'{det50*100:.1f}%', _pct_bar(det50))
    t.add_row('Laser visible (50 ep)',    f'{las50*100:.1f}%', _pct_bar(las50))
    t.add_row('Lookahead U moy.',
              f'{_smooth([_float(r,"visual_lookahead_u") for r in ep_rows],20):.3f}', '')
    t.add_row('Lookahead V moy.',
              f'{_smooth([_float(r,"visual_lookahead_v") for r in ep_rows],20):.3f}', '')
    t.add_row('Visual progress moy.',
              f'{_smooth([_float(r,"visual_progress") for r in ep_rows],20)*100:.1f}%', '')

    return Panel(t, title='[bold blue]■ KLT Caméra[/bold blue]',
                 border_style='blue')


def _panel_mgd(ep_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Métrique MGD', style='bold green', width=26)
    t.add_column('Valeur', width=16)
    t.add_column('Tendance', width=24)

    if not ep_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold green]MGD/Cinématique[/bold green]')

    yosh  = [_float(r, 'yoshikawa_wall')  for r in ep_rows]
    conds = [_float(r, 'cond_wall')       for r in ep_rows]
    sings = [_float(r, 'sing_penalty', 0) for r in ep_rows]

    yosh50  = _smooth(yosh, 50)
    cond50  = _smooth(conds, 50)
    sing50  = _smooth(sings, 50)

    yosh_ok = yosh50 > 0.04
    t.add_row('Yoshikawa w (50 ep)',
              f'[{"green" if yosh_ok else "red"}]{yosh50:.4f}[/{"green" if yosh_ok else "red"}]',
              _pct_bar(yosh50 / 0.15, color_ok='green', color_bad='red'))
    t.add_row('Conditionnement mur',
              f'{cond50:.1f}',
              '[green]OK[/green]' if cond50 < 50 else '[red]élevé[/red]')
    t.add_row('Pén. singularité moy.', f'{sing50:.4f}', '')
    t.add_row('Seuil W_MIN', '0.04', '')

    return Panel(t, title='[bold green]■ MGD / Singularités[/bold green]',
                 border_style='green')


def _panel_lqr(ep_rows: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column('Métrique LQR', style='bold red', width=26)
    t.add_column('Valeur', width=16)
    t.add_column('Tendance', width=24)

    if not ep_rows:
        return Panel('[dim]En attente…[/dim]', title='[bold red]LQR / Commande[/bold red]')

    raws  = [_float(r, 'cmd_raw_norm',  0.0) for r in ep_rows]
    lqrs  = [_float(r, 'cmd_lqr_norm',  0.0) for r in ep_rows]
    nulls = [_float(r, 'cmd_null_norm', 0.0) for r in ep_rows]
    outs  = [_float(r, 'cmd_out_norm',  0.0) for r in ep_rows]

    raw50  = _smooth(raws, 50)
    lqr50  = _smooth(lqrs, 50)
    null50 = _smooth(nulls, 50)
    out50  = _smooth(outs, 50)

    t.add_row('‖cmd_raw‖  (RL brut)',       f'{raw50:.4f} rad/s', '')
    t.add_row('‖cmd_lqr‖  (après Riccati)', f'{lqr50:.4f} rad/s', '')
    t.add_row('‖cmd_null‖ (noyau Jac.)',    f'{null50:.4f} rad/s', '')
    t.add_row('‖cmd_out‖  (sortie finale)', f'{out50:.4f} rad/s',
              _pct_bar(1.0 - min(out50 / 2.0, 1.0)))

    return Panel(t, title='[bold red]■ LQR / Commande[/bold red]',
                 border_style='red')


def _panel_snapshots(metrics_dir: Path) -> Panel:
    snap_csv = metrics_dir / 'trajectory_snapshots' / 'snapshot_rmse.csv'
    if not snap_csv.exists():
        return Panel('[dim]Aucun snapshot encore.[/dim]',
                     title='[bold white]Snapshots Trajectoires[/bold white]')
    rows = _read_tail(snap_csv, 15)
    if not rows:
        return Panel('[dim]Aucun snapshot.[/dim]',
                     title='[bold white]Snapshots Trajectoires[/bold white]')
    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column('#',       style='dim',         width=4)
    t.add_column('Step',    style='bold',        width=10)
    t.add_column('RMSE cm', style='bold',        width=10)
    t.add_column('Prog.',                         width=8)
    t.add_column('Succès',                        width=8)
    for r in rows[-10:]:
        step   = _int(r, 'timestep')
        snap   = _int(r, 'snapshot')
        rmse   = _float(r, 'rmse_cm')
        prog   = _float(r, 'progress')
        ok     = _int(r, 'success')
        rmse_s = f'{rmse:.1f}' if not (rmse != rmse) else 'N/A'
        color  = 'green' if ok else ('yellow' if prog > 0.5 else 'red')
        t.add_row(str(snap), f'{step:,}',
                  f'[{color}]{rmse_s}[/{color}]',
                  f'{prog*100:.0f}%',
                  '[green]✓[/green]' if ok else '[red]✗[/red]')
    return Panel(t, title='[bold white]Snapshots Trajectoires (10k steps)[/bold white]',
                 border_style='white')


def _panel_header(ep_rows: list[dict], total: int) -> Panel:
    if not ep_rows:
        return Panel('[dim]Démarrage training…[/dim]',
                     title='[bold]UR7e Line Follower — SAC Training Monitor[/bold]')
    step = _int(ep_rows[-1], 'timestep')
    ep   = _int(ep_rows[-1], 'episode')
    total = max(int(total), step, 1)
    pct = min(step / total, 1.0)
    bar_w = 50
    filled = int(pct * bar_w)
    bar = '█' * filled + '░' * (bar_w - filled)
    succ50 = _smooth([_float(r, 'success') for r in ep_rows], 50)
    prog50 = _smooth([_float(r, 'progress') for r in ep_rows], 50)
    rew50  = _smooth([_float(r, 'episode_reward') for r in ep_rows], 50)

    txt = (f'[bold cyan]Step {step:>7,} / {total:,}[/bold cyan]   '
           f'[{bar}] [bold]{pct*100:.1f}%[/bold]\n'
           f'Épisodes : [bold]{ep}[/bold]   '
           f'Succès : [bold green]{succ50*100:.0f}%[/bold green]   '
           f'Progression : [bold yellow]{prog50*100:.0f}%[/bold yellow]   '
           f'Reward : [bold]{rew50:+.2f}[/bold]   '
           f'[dim]{time.strftime("%H:%M:%S")}[/dim]')
    return Panel(txt, title='[bold]■ UR7e Line Follower — SAC Training Monitor[/bold]',
                 border_style='bright_white')


# ── Boucle principale ──────────────────────────────────────────────────────────

def build_layout(ep_rows, win_rows, up_rows, metrics_dir: Path, total: int) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=5),
        Layout(name='body'),
        Layout(name='footer', size=14),
    )
    layout['body'].split_row(
        Layout(name='left'),
        Layout(name='right'),
    )
    layout['left'].split_column(
        Layout(name='rl',        ratio=3),
        Layout(name='reward_bk', ratio=4),
    )
    layout['right'].split_column(
        Layout(name='sac',    ratio=2),
        Layout(name='kalman', ratio=2),
        Layout(name='klt',    ratio=2),
    )
    layout['header'].update(_panel_header(ep_rows, total))
    layout['rl'].update(_panel_rl(ep_rows, win_rows))
    layout['reward_bk'].update(_panel_reward_breakdown(ep_rows))
    layout['sac'].update(_panel_sac_internals(up_rows))
    layout['kalman'].update(_panel_kalman(ep_rows))
    layout['klt'].update(_panel_klt(ep_rows))

    # Bas : MGD + LQR + snapshots
    layout['footer'].split_row(
        Layout(_panel_mgd(ep_rows),       name='mgd'),
        Layout(_panel_lqr(ep_rows),       name='lqr'),
        Layout(_panel_snapshots(metrics_dir), name='snaps'),
    )
    return layout


def run(refresh: float = 5.0, metrics: str | None = None):
    ep_path  = None
    win_path = None
    up_path  = None

    metrics_dir = _resolve_metrics_dir(metrics)
    total = _requested_total(metrics_dir)
    console.print('[bold green]Training Monitor démarré. Ctrl+C pour quitter.[/bold green]')
    console.print(f'[dim]Dossier métriques : {metrics_dir}[/dim]\n')

    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        while True:
            # Chercher les CSV les plus récents
            new_ep  = _latest('train_episodes_*.csv', metrics_dir)
            new_win = _latest('train_windows_*.csv',  metrics_dir)
            new_up  = _latest('train_updates_*.csv',  metrics_dir)
            if new_ep:  ep_path  = new_ep
            if new_win: win_path = new_win
            if new_up:  up_path  = new_up

            ep_rows  = _read_tail(ep_path,  200) if ep_path  else []
            win_rows = _read_tail(win_path, 100) if win_path else []
            up_rows  = _read_tail(up_path,  100) if up_path  else []

            try:
                live.update(build_layout(ep_rows, win_rows, up_rows, metrics_dir, total))
            except Exception:
                pass
            time.sleep(refresh)


def main():
    parser = argparse.ArgumentParser(description='Dashboard terminal training UR7e')
    parser.add_argument('--refresh', type=float, default=5.0,
                        help='Intervalle de rafraîchissement en secondes (défaut: 5)')
    parser.add_argument('--metrics', default=None,
                        help='Dossier metrics ou dossier de run; défaut: dernier run')
    args = parser.parse_args()
    try:
        run(refresh=args.refresh, metrics=args.metrics)
    except KeyboardInterrupt:
        console.print('\n[bold red]Arrêt du monitor.[/bold red]')


if __name__ == '__main__':
    main()
