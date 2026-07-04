"""Entraînement SAC + logging CSV propre pour UR7e line follower."""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import pickle
import shutil
import faulthandler
from collections import deque
from pathlib import Path

faulthandler.enable()

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .env import UR7eLineFollowerEnv, OBSERVATION_SCHEMA_VERSION, OBSERVATION_SPACE_DIM
from .control import ACTION_SPACE_DIM, CONTROL_SCHEMA_VERSION, MAX_WALL_SPEED_M_S
from .reward import DEFAULT_REWARD_PROFILE, REWARD_PROFILES

torch.set_num_threads(2)

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
TB_LOG_DIR = DATA_DIR / 'tb_logs'
METRICS_DIR = DATA_DIR / 'metrics'
for d in (DATA_DIR, CHECKPOINT_DIR, TB_LOG_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 3e-4        # standard SAC fresh start
BATCH_SIZE = 256
BUFFER_SIZE = 300_000
LEARNING_STARTS = 5_000
GAMMA = 0.99
TAU = 0.005
TRAIN_FREQ = 1              # update à chaque step = apprentissage plus réactif
GRADIENT_STEPS = 1          # 1 update/step = plus stable
NET_ARCH = [256, 256]
SAVE_FREQ = 5_000
LOG_FREQ = 200
ENT_COEF_RESET = 0.03

N_DEMOS_DEFAULT = 0
DEMO_NOISE = 0.02
DEMO_MAX_ATTEMPTS_FACTOR = 5
ACTIVE_RUN_CONFIG: dict = {}


def _scripted_action(env: UR7eLineFollowerEnv, rng: np.random.Generator,
                     kp: float = 2.5) -> np.ndarray:
    """Geometric expert used only to seed the replay buffer."""
    dot = env.node.get_laser_dot()
    if dot is None:
        return np.zeros(ACTION_SPACE_DIM, dtype=np.float32)
    lookahead_idx = min(max(env._wp_idx, 1) + 2, env._n_wp - 1)
    target = np.asarray(env.waypoints[lookahead_idx], dtype=np.float64)
    error = target - np.asarray(dot, dtype=np.float64)
    effective_max = float(
        env._action_speed_scale() * MAX_WALL_SPEED_M_S)
    effective_max = max(effective_max, 1e-3)
    desired = float(kp) * error
    norm = float(np.linalg.norm(desired))
    if norm > effective_max:
        desired *= effective_max / norm
    action = desired / effective_max
    action += rng.normal(0.0, DEMO_NOISE, size=ACTION_SPACE_DIM)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def generate_demos(env, n_demos: int, save_path: Path, seed: int = 0) -> list:
    """Generate successful expert episodes with a bounded number of attempts."""
    if n_demos <= 0:
        return []
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    episodes: list = []
    if save_path.exists():
        try:
            with open(save_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, list):
                episodes = cached[:n_demos]
                print(f'[demos] {len(episodes)} épisodes chargés depuis {save_path}')
        except Exception as exc:
            print(f'[demos] cache ignoré ({exc})')
            episodes = []
    rng = np.random.default_rng(seed + len(episodes))
    max_attempts = max(n_demos, n_demos * DEMO_MAX_ATTEMPTS_FACTOR)
    attempts = 0
    while len(episodes) < n_demos and attempts < max_attempts:
        attempts += 1
        reset_kwargs = {'seed': int(seed)} if attempts == 1 else {}
        obs, _ = env.reset(**reset_kwargs)
        transitions = []
        success = False
        max_ep_steps = int(env.unwrapped._max_steps)
        for _ in range(max_ep_steps):
            action = _scripted_action(env.unwrapped, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            transitions.append((
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                done,
                {'is_success': bool(info.get('is_success', False)),
                 'TimeLimit.truncated': bool(truncated and not terminated)},
            ))
            obs = next_obs
            if done:
                success = bool(info.get('is_success', False))
                break
        if success:
            episodes.append(transitions)
            print(f'[demos] {len(episodes)}/{n_demos} ✓ ({len(transitions)} steps)')
            with open(save_path, 'wb') as f:
                pickle.dump(episodes, f)
        else:
            print(f'[demos] tentative {attempts}/{max_attempts} non réussie')
    if len(episodes) < n_demos:
        print(f'[demos] AVERTISSEMENT: seulement {len(episodes)}/{n_demos} démonstrations valides')
    return episodes


def inject_demos(model: SAC, episodes: list, repeat: int = 2) -> int:
    """Insert expert transitions into the standard SAC replay buffer."""
    count = 0
    for _ in range(max(1, int(repeat))):
        for episode in episodes:
            for obs, action, reward, next_obs, done, info in episode:
                model.replay_buffer.add(
                    obs=np.asarray(obs, dtype=np.float32)[None, :],
                    next_obs=np.asarray(next_obs, dtype=np.float32)[None, :],
                    action=np.asarray(action, dtype=np.float32)[None, :],
                    reward=np.array([reward], dtype=np.float32),
                    done=np.array([done], dtype=np.float32),
                    infos=[info],
                )
                count += 1
    print(f'[demos] {count} transitions expertes injectées dans le replay buffer')
    return count


def parse_ent_coef(value: str | float):
    """Parse a Stable-Baselines3 entropy coefficient CLI value."""
    if isinstance(value, (float, int)):
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError("ent_coef doit être strictement positif")
        return parsed
    text = str(value).strip().lower()
    if text == "auto" or text.startswith("auto_"):
        return text
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ent-coef attend 'auto', 'auto_0.05' ou un float positif") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("--ent-coef doit être strictement positif")
    return parsed


class TripwireCallback(BaseCallback):
    """Early learning trend monitor for the mandatory Phase-1 run.

    It prints terminal-episode medians and can optionally stop a run when, after
    a configurable minimum budget, neither success nor a meaningful distance
    improvement is observed.  Automatic stopping is disabled by default.
    """

    def __init__(self, every: int = 1000, window: int = 50,
                 early_stop: bool = False, min_steps: int = 10_000,
                 min_relative_improvement: float = 0.10, verbose: int = 1):
        super().__init__(verbose)
        self.every = max(1, int(every))
        self.window = max(5, int(window))
        self.early_stop = bool(early_stop)
        self.min_steps = max(self.every, int(min_steps))
        self.min_relative_improvement = float(min_relative_improvement)
        self._dist = deque(maxlen=self.window)
        self._off = deque(maxlen=self.window)
        self._success = deque(maxlen=self.window)
        self._progress = deque(maxlen=self.window)
        self._baseline_distance: float | None = None
        self.stopped_early = False

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue
            distance = float(info.get('ordered_deviation_mean', np.nan))
            if np.isfinite(distance):
                self._dist.append(distance)
            offwall = float(info.get('offwall_ratio', np.nan))
            if np.isfinite(offwall):
                self._off.append(offwall)
            self._success.append(1.0 if info.get('is_success', False) else 0.0)
            self._progress.append(float(info.get('progress', 0.0)))

        if self.num_timesteps % self.every != 0 or not self._dist:
            return True

        distance = float(np.median(self._dist))
        offwall = float(np.mean(self._off)) if self._off else float('nan')
        success = float(np.mean(self._success)) if self._success else 0.0
        progress = float(np.mean(self._progress)) if self._progress else 0.0
        if self._baseline_distance is None and len(self._dist) >= 5:
            self._baseline_distance = distance
        improvement = 0.0
        if self._baseline_distance and self._baseline_distance > 1e-9:
            improvement = 1.0 - distance / self._baseline_distance
        print(
            f'[tripwire] t={self.num_timesteps:>7} | dist_med={distance*100:5.1f}cm '
            f'| amélioration={improvement*100:+5.1f}% | offwall={offwall*100:4.0f}% '
            f'| prog={progress*100:4.0f}% | succès={success*100:4.0f}%')

        if (self.early_stop and self.num_timesteps >= self.min_steps
                and success <= 0.0
                and improvement < self.min_relative_improvement):
            self.stopped_early = True
            print(
                '[tripwire] NO-GO automatique : aucune réussite et amélioration '
                f'< {self.min_relative_improvement*100:.0f}% après '
                f'{self.num_timesteps} steps. Sauvegarde puis arrêt.')
            return False
        return True


class MetricsCallback(BaseCallback):
    """CSV détaillés : épisodes, fenêtres d'entraînement, pertes SAC."""

    def __init__(self, log_freq: int = LOG_FREQ, metrics_dir: Path = METRICS_DIR):
        super().__init__()
        self.log_freq = int(log_freq)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = time.strftime('%Y%m%d_%H%M%S')
        self.ep_path = self.metrics_dir / f'train_episodes_{self.run_tag}.csv'
        self.win_path = self.metrics_dir / f'train_windows_{self.run_tag}.csv'
        self.up_path = self.metrics_dir / f'train_updates_{self.run_tag}.csv'
        self._ep_file = open(self.ep_path, 'w', newline='')
        self._win_file = open(self.win_path, 'w', newline='')
        self._up_file = open(self.up_path, 'w', newline='')
        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=[
            'timestep', 'episode', 'episode_reward', 'episode_length',
            'success', 'progress', 'ep_rmse_m', 'recent_rmse_m', 'deviation_mean_m', 'deviation_max_m',
            'ordered_deviation_mean_m', 'offwall_ratio', 'traj_count',
            'yoshikawa_wall', 'cond_wall', 'sing_penalty',
            'cmd_raw_norm', 'cmd_lqr_norm', 'cmd_null_norm', 'cmd_out_norm', 'pulse_duration_s',
            'ekf_sigma_mean_m', 'ekf_nis', 'cam_detected', 'cam_laser_visible',
            'camera_age_s', 'detection_age_s', 'fresh_camera_frame',
            'reward_dist_ordered', 'reward_sing_penalty', 'reward_cmd_penalty',
            'reward_offwall_penalty', 'reward_waypoint_bonus', 'reward_completion_bonus',
            'reward_progress_reward', 'reward_vision_penalty', 'reward_stale_camera_penalty',
            'reward_action_delta_penalty', 'reward_stagnation_penalty', 'reward_record_bonus',
            'visual_lookahead_u', 'visual_lookahead_v', 'visual_progress',
            'stagnation_steps', 'curriculum_level', 'curriculum_success_rate', 'action_speed_scale',
            'physics_step_timeout', 'offwall_timeout', 'stagnation_timeout',
            'camera_transport_timeout', 'line_lost_timeout', 'laser_lost_timeout',
        ])
        self._win_writer = csv.DictWriter(self._win_file, fieldnames=[
            'timestep', 'episodes_in_window', 'reward_mean', 'reward_std',
            'success_rate', 'progress_mean', 'rmse_mean_m', 'recent_rmse_mean_m', 'rmse_p50_m', 'rmse_p95_m',
            'deviation_mean_m', 'offwall_ratio_mean', 'cmd_out_norm_mean',
            'ekf_sigma_mean_m', 'yoshikawa_wall_mean', 'cond_wall_median',
        ])
        self._up_writer = csv.DictWriter(self._up_file, fieldnames=[
            'timestep', 'actor_loss', 'critic_loss', 'ent_coef', 'ent_coef_loss',
            'learning_rate', 'n_updates', 'buffer_size',
        ])
        for w in (self._ep_writer, self._win_writer, self._up_writer):
            w.writeheader()
        self._episode = 0
        self._last_log = 0
        self._window_rows: list[dict] = []
        print(f'[metrics] épisodes : {self.ep_path}')
        print(f'[metrics] fenêtres : {self.win_path}')
        print(f'[metrics] updates  : {self.up_path}')

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            ep_data = info.get('episode', {}) if isinstance(info, dict) else {}
            row = {
                'timestep': self.num_timesteps,
                'episode': self._episode + 1,
                'episode_reward': float(ep_data.get('r', np.nan)),
                'episode_length': int(ep_data.get('l', 0)),
                'success': int(info.get('is_success', False)),
                'progress': float(info.get('progress', 0.0)),
                'ep_rmse_m': float(info.get('ep_rmse', np.nan)),
                'recent_rmse_m': float(info.get('recent_rmse', np.nan)),
                'deviation_mean_m': float(info.get('deviation_mean', np.nan)),
                'deviation_max_m': float(info.get('deviation_max', np.nan)),
                'ordered_deviation_mean_m': float(info.get('ordered_deviation_mean', np.nan)),
                'offwall_ratio': float(info.get('offwall_ratio', np.nan)),
                'traj_count': int(info.get('traj_count', 0)),
                'yoshikawa_wall': float(info.get('yoshikawa_w', np.nan)),
                'cond_wall': float(info.get('cond_wall', np.nan)),
                'sing_penalty': float(info.get('sing_penalty', np.nan)),
                'cmd_raw_norm': float(info.get('cmd_raw_norm', np.nan)),
                'cmd_lqr_norm': float(info.get('cmd_lqr_norm', np.nan)),
                'cmd_null_norm': float(info.get('cmd_null_norm', np.nan)),
                'cmd_out_norm': float(info.get('cmd_out_norm', np.nan)),
                'pulse_duration_s': float(info.get('pulse_duration_s', np.nan)),
                'ekf_sigma_mean_m': float(info.get('ekf_sigma_mean', np.nan)),
                'ekf_nis': float(info.get('ekf_nis', np.nan)),
                'cam_detected': float(info.get('cam_detected', np.nan)),
                'cam_laser_visible': float(info.get('cam_laser_visible', np.nan)),
                'camera_age_s': float(info.get('camera_age_s', np.nan)),
                'detection_age_s': float(info.get('detection_age_s', np.nan)),
                'fresh_camera_frame': int(info.get('fresh_camera_frame', False)),
                'reward_dist_ordered': float(info.get('reward_dist_ordered', np.nan)),
                'reward_sing_penalty': float(info.get('reward_sing_penalty', np.nan)),
                'reward_cmd_penalty': float(info.get('reward_cmd_penalty', np.nan)),
                'reward_offwall_penalty': float(info.get('reward_offwall_penalty', np.nan)),
                'reward_waypoint_bonus': float(info.get('reward_waypoint_bonus', np.nan)),
                'reward_completion_bonus': float(info.get('reward_completion_bonus', np.nan)),
                'reward_progress_reward': float(info.get('reward_progress_reward', np.nan)),
                'reward_vision_penalty': float(info.get('reward_vision_penalty', np.nan)),
                'reward_stale_camera_penalty': float(info.get('reward_stale_camera_penalty', np.nan)),
                'reward_action_delta_penalty': float(info.get('reward_action_delta_penalty', np.nan)),
                'reward_stagnation_penalty': float(info.get('reward_stagnation_penalty', np.nan)),
                'reward_record_bonus': float(info.get('reward_record_bonus', np.nan)),
                'visual_lookahead_u': float(info.get('visual_lookahead_u', np.nan)),
                'visual_lookahead_v': float(info.get('visual_lookahead_v', np.nan)),
                'visual_progress': float(info.get('visual_progress', np.nan)),
                'stagnation_steps': int(info.get('stagnation_steps', 0)),
                'curriculum_level': int(info.get('curriculum_level', 0)),
                'curriculum_success_rate': float(info.get('curriculum_success_rate', 0.0)),
                'action_speed_scale': float(info.get('action_speed_scale', 1.0)),
                'physics_step_timeout': int(info.get('physics_step_timeout', False)),
                'offwall_timeout': int(info.get('offwall_timeout', False)),
                'stagnation_timeout': int(info.get('stagnation_timeout', False)),
                'camera_transport_timeout': int(info.get('camera_transport_timeout', False)),
                'line_lost_timeout': int(info.get('line_lost_timeout', False)),
                'laser_lost_timeout': int(info.get('laser_lost_timeout', False)),
            }
            self._ep_writer.writerow(row)
            self._window_rows.append(row)
            self._episode += 1

        if self.num_timesteps - self._last_log >= self.log_freq:
            self._write_window()
            self._write_update()
            self._last_log = self.num_timesteps
        return True

    @staticmethod
    def _nanmean(rows, key, default=np.nan):
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])], dtype=float)
        return float(np.mean(vals)) if len(vals) else default

    def _write_window(self):
        if not self._window_rows:
            return
        rows = self._window_rows
        rewards = np.array([r['episode_reward'] for r in rows if np.isfinite(r['episode_reward'])], dtype=float)
        rmses = np.array([r['ep_rmse_m'] for r in rows if np.isfinite(r['ep_rmse_m'])], dtype=float)
        conds = np.array([r['cond_wall'] for r in rows if np.isfinite(r['cond_wall'])], dtype=float)
        recent_rmses = np.array([r['recent_rmse_m'] for r in rows if np.isfinite(r['recent_rmse_m'])], dtype=float)
        row = {
            'timestep': self.num_timesteps,
            'episodes_in_window': len(rows),
            'reward_mean': float(np.mean(rewards)) if len(rewards) else np.nan,
            'reward_std': float(np.std(rewards)) if len(rewards) else np.nan,
            'success_rate': float(np.mean([r['success'] for r in rows])),
            'progress_mean': self._nanmean(rows, 'progress'),
            'rmse_mean_m': float(np.mean(rmses)) if len(rmses) else np.nan,
            'recent_rmse_mean_m': float(np.mean(recent_rmses)) if len(recent_rmses) else np.nan,
            'rmse_p50_m': float(np.percentile(rmses, 50)) if len(rmses) else np.nan,
            'rmse_p95_m': float(np.percentile(rmses, 95)) if len(rmses) else np.nan,
            'deviation_mean_m': self._nanmean(rows, 'deviation_mean_m'),
            'offwall_ratio_mean': self._nanmean(rows, 'offwall_ratio'),
            'cmd_out_norm_mean': self._nanmean(rows, 'cmd_out_norm'),
            'ekf_sigma_mean_m': self._nanmean(rows, 'ekf_sigma_mean_m'),
            'yoshikawa_wall_mean': self._nanmean(rows, 'yoshikawa_wall'),
            'cond_wall_median': float(np.median(conds)) if len(conds) else np.nan,
        }
        self._win_writer.writerow(row)
        self._ep_file.flush(); self._win_file.flush()
        print(f"[train] step={self.num_timesteps:,} ep={self._episode} "
              f"rew={row['reward_mean']:+.2f} ok={row['success_rate']*100:.0f}% "
              f"prog={row['progress_mean']*100:.0f}% rmse={row['rmse_mean_m']*100:.1f}cm")
        self._window_rows.clear()

    def _write_update(self):
        log = self.model.logger.name_to_value
        try:
            rb_size = self.model.replay_buffer.size()
        except Exception:
            rb_size = np.nan
        row = {
            'timestep': self.num_timesteps,
            'actor_loss': log.get('train/actor_loss', np.nan),
            'critic_loss': log.get('train/critic_loss', np.nan),
            'ent_coef': log.get('train/ent_coef', np.nan),
            'ent_coef_loss': log.get('train/ent_coef_loss', np.nan),
            'learning_rate': log.get('train/learning_rate', np.nan),
            'n_updates': log.get('train/n_updates', np.nan),
            'buffer_size': rb_size,
        }
        self._up_writer.writerow(row)
        self._up_file.flush()

    def _on_training_end(self):
        self._write_window()
        self._write_update()
        self._ep_file.close(); self._win_file.close(); self._up_file.close()


SNAPSHOT_FREQ = 10_000  # snapshot trajectoire toutes les N steps


class TrajectorySnapshotCallback(BaseCallback):
    """Sauvegarde toutes les SNAPSHOT_FREQ steps un PNG et deux CSV comparant
    la trajectoire laser réelle vs la cible bleue — des snapshots réguliers indépendamment du budget total."""

    def __init__(self, freq: int = SNAPSHOT_FREQ, out_dir: Path = METRICS_DIR):
        super().__init__()
        self.freq = freq
        self.snap_dir = Path(out_dir) / 'trajectory_snapshots'
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._last = 0
        self._pending: tuple | None = None  # (laser_path, waypoints, timestep)
        self._count = 0
        self._rmse_log_path = self.snap_dir / 'snapshot_rmse.csv'
        with open(self._rmse_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['timestep', 'snapshot', 'rmse_cm', 'progress', 'success'])
        print(f'[snapshots] dossier : {self.snap_dir}')

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done and isinstance(info, dict) and 'laser_path' in info:
                self._pending = (
                    info['laser_path'],
                    info.get('target_waypoints'),
                    self.num_timesteps,
                    float(info.get('progress', 0.0)),
                    bool(info.get('is_success', False)),
                )
        if self.num_timesteps - self._last >= self.freq:
            if self._pending is not None:
                self._save(*self._pending)
                self._pending = None
            self._last = self.num_timesteps
        return True

    def _save(self, laser: np.ndarray, target: np.ndarray | None,
              step: int, progress: float, success: bool):
        self._count += 1
        tag = f'snap_{step:07d}'

        # CSV bruts
        if laser is not None and len(laser) > 0:
            np.savetxt(self.snap_dir / f'{tag}_laser.csv', laser,
                       delimiter=',', header='y,z', comments='')
        if target is not None:
            np.savetxt(self.snap_dir / f'{tag}_target.csv', target,
                       delimiter=',', header='y,z', comments='')

        # RMSE par distance point-à-ligne la plus proche
        rmse_cm = float('nan')
        if laser is not None and len(laser) > 0 and target is not None and len(target) > 0:
            diffs = laser[:, None, :] - target[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            rmse_cm = float(np.sqrt(np.mean(np.min(dists, axis=1) ** 2))) * 100.0

        with open(self._rmse_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([step, self._count, f'{rmse_cm:.3f}', f'{progress:.3f}', int(success)])

        # Figure 2D trajectoire
        fig, ax = plt.subplots(figsize=(6, 6))
        if target is not None and len(target) > 0:
            ax.plot(target[:, 0], target[:, 1], 'b-', lw=2.5, label='Cible (bleue)', zorder=2)
            ax.scatter(target[0, 0], target[0, 1], c='#e74c3c', s=100, zorder=5, label='Départ')
            ax.scatter(target[-1, 0], target[-1, 1], c='#2ecc71', s=100, zorder=5, label='Arrivée')
        if laser is not None and len(laser) > 0:
            ax.plot(laser[:, 0], laser[:, 1], 'r-', lw=1.5, alpha=0.85, label='Laser RL', zorder=3)
        title = (f'Trajectoire — step {step:,}  |  '
                 f'{"✓ SUCCÈS" if success else f"prog={progress*100:.0f}%"}')
        if not np.isnan(rmse_cm):
            title += f'\nRMSE = {rmse_cm:.1f} cm'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.snap_dir / f'{tag}_traj.png', dpi=120)
        plt.close(fig)
        print(f'[snapshot #{self._count}] step {step:,} | RMSE={rmse_cm:.1f}cm | '
              f'prog={progress*100:.0f}% → {tag}_traj.png')

    def _on_training_end(self):
        if self._pending is not None:
            self._save(*self._pending)


class SchemaCheckpointCallback(CheckpointCallback):
    """Checkpoint SB3 accompagné systématiquement de son fichier .meta.json."""

    def _on_step(self) -> bool:
        should_save = self.n_calls % self.save_freq == 0
        keep_training = super()._on_step()
        if should_save:
            checkpoint_zip = Path(self._checkpoint_path(extension='zip'))
            _save_meta(Path(str(checkpoint_zip).removesuffix('.zip')))
        return keep_training


class BestTrainingModelCallback(BaseCallback):
    """Sauvegarde le meilleur modèle à partir d'une moyenne d'épisodes réels.

    Aucun second environnement Gazebo n'est créé : cela évite que l'évaluation
    et l'entraînement commandent simultanément le même robot.
    """

    def __init__(self, save_dir: Path = CHECKPOINT_DIR, window: int = 10,
                 min_episodes: int = 5):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.window = max(1, int(window))
        self.min_episodes = max(1, int(min_episodes))
        self.scores = deque(maxlen=self.window)
        self.best_score = -float('inf')
        self.episodes = 0

    @staticmethod
    def _score(info: dict) -> float:
        success = 1.0 if info.get('is_success', False) else 0.0
        progress = float(info.get('progress', 0.0))
        rmse = float(info.get('ep_rmse', 0.50))
        smooth_penalty = abs(float(info.get('reward_action_delta_penalty', 0.0)))
        return 100.0 * success + 20.0 * progress - 25.0 * min(rmse, 0.50) - smooth_penalty

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue
            self.episodes += 1
            self.scores.append(self._score(info))
            if self.episodes < self.min_episodes or len(self.scores) < self.min_episodes:
                continue
            mean_score = float(np.mean(self.scores))
            if mean_score <= self.best_score:
                continue
            self.best_score = mean_score
            path = self.save_dir / 'best_model'
            self.model.save(str(path))
            _save_meta(path)
            print(f'[best] modèle sauvegardé | score={mean_score:.3f} | épisodes={self.episodes}')
        return True


def _check_schema(model_path: str) -> None:
    """Vérifie la compatibilité du checkpoint avec le schéma V5 courant."""
    meta_path = Path(str(model_path).removesuffix('.zip') + '.meta.json')
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[train] Pas de fichier méta : {meta_path}\n"
            f"Les checkpoints V1 (28D) sont incompatibles avec le schéma "
            f"V{OBSERVATION_SCHEMA_VERSION} ({OBSERVATION_SPACE_DIM}D). Réentraîner depuis zéro."
        )
    meta = json.loads(meta_path.read_text())
    saved_v = meta.get('observation_schema_version', 1)
    if saved_v != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"[train] Schéma V{saved_v} sauvegardé ≠ V{OBSERVATION_SCHEMA_VERSION} courant.\n"
            f"Réentraîner depuis zéro ou migrer le checkpoint."
        )
    if int(meta.get('obs_shape', -1)) != OBSERVATION_SPACE_DIM:
        raise ValueError(
            f"[train] meta obs_shape={meta.get('obs_shape')} ≠ {OBSERVATION_SPACE_DIM}."
        )
    if int(meta.get('action_shape', -1)) != ACTION_SPACE_DIM:
        raise ValueError(
            f"[train] meta action_shape={meta.get('action_shape')} ≠ {ACTION_SPACE_DIM}. "
            "Les checkpoints V2.1 à commande articulaire 6D sont incompatibles."
        )
    if int(meta.get('control_schema_version', -1)) != CONTROL_SCHEMA_VERSION:
        raise ValueError(
            f"[train] control_schema_version={meta.get('control_schema_version')} "
            f"≠ {CONTROL_SCHEMA_VERSION}."
        )


def _load_model_checked(model_path: str, env) -> 'SAC':
    """Charge un modèle SAC avec vérification dure du schéma et des formes."""
    _check_schema(model_path)
    # Charger sans env d'abord pour pouvoir inspecter les espaces
    model = SAC.load(model_path, env=None)
    obs_shape = tuple(model.observation_space.shape)
    act_shape = tuple(model.action_space.shape)
    if obs_shape != (OBSERVATION_SPACE_DIM,):
        raise ValueError(
            f"[train] obs_space.shape={obs_shape} ≠ ({OBSERVATION_SPACE_DIM},). "
            f"Checkpoint incompatible."
        )
    if act_shape != (ACTION_SPACE_DIM,):
        raise ValueError(
            f"[train] action_space.shape={act_shape} ≠ ({ACTION_SPACE_DIM},). "
            "Checkpoint incompatible avec la commande cartésienne V2.2."
        )
    model.set_env(env)
    return model


def _save_meta(model_path: Path) -> None:
    meta = {
        'observation_schema_version': OBSERVATION_SCHEMA_VERSION,
        'obs_shape': OBSERVATION_SPACE_DIM,
        'action_shape': ACTION_SPACE_DIM,
        'control_schema_version': CONTROL_SCHEMA_VERSION,
        'control_mode': 'wall_velocity_yz_deterministic_pulse',
        'camera_schema_length': 7,
        'camera_schema_semantics': 'offset + confidence + directed_tangent_to_green_flag',
        'visual_guidance_length': 3,
        'visual_guidance_semantics': 'lookahead_du + lookahead_dv + visual_progress',
        'rl_features': [
            'deterministic_pulse', 'full_range_tracking_reward',
            'ordered_progress_fix', 'ema_state_observed', 'guided_reset',
            'competence_curriculum', 'recoverable_recent_rmse_success',
            'expert_demo_injection', 'per_run_logging',
        ],
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        **ACTIVE_RUN_CONFIG,
    }
    meta_path = model_path.parent / f'{model_path.name}.meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[train] méta sauvegardé : {meta_path}')


def make_env(line_shape: str, random_trajectories: bool, headless: bool = True,
             trials_per_trajectory: int = 5, curriculum: bool = True,
             training_profile: str = 'realistic',
             observation_mode: str = 'real',
             sensor_noise: bool = True,
             reward_profile: str | None = None):
    if reward_profile is None:
        reward_profile = _reward_profile_from_meta(model_path)
    env = UR7eLineFollowerEnv(
        line_shape=line_shape,
        sensor_noise=sensor_noise,
        random_trajectories=random_trajectories,
        update_dot_visual=False,
        observation_mode=observation_mode,
        trials_per_trajectory=trials_per_trajectory,
        curriculum=curriculum,
        training_profile=training_profile,
        guided_reset=True,
        deterministic_pulse=True,
        reward_profile=reward_profile,
    )
    return Monitor(env)


def train(line_shape: str = 's_curve', random_trajectories: bool = True,
          total_timesteps: int = TOTAL_TIMESTEPS, headless: bool = True,
          trials_per_trajectory: int = 5, curriculum: bool = True,
          training_profile: str = 'realistic', observation_mode: str = 'real',
          n_demos: int = N_DEMOS_DEFAULT, seed: int = 0,
          run_env_check: bool = False, save_replay_buffer: bool = False,
          reward_profile: str = DEFAULT_REWARD_PROFILE,
          ent_coef=ENT_COEF_RESET, tripwire_every: int = 1000,
          tripwire_early_stop: bool = False):
    global ACTIVE_RUN_CONFIG
    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f'reward_profile invalide: {reward_profile!r}')
    ent_coef = parse_ent_coef(ent_coef)

    if training_profile == 'minimal_straight_line_debug':
        random_trajectories = False
        curriculum = False
        observation_mode = 'privileged_debug'
        trials_per_trajectory = 10_000
        sensor_noise = False
    else:
        sensor_noise = True

    run_tag = time.strftime('%Y%m%d_%H%M%S')
    run_dir = DATA_DIR / 'runs' / f'{run_tag}_{training_profile}'
    run_checkpoint_dir = run_dir / 'checkpoints'
    run_metrics_dir = run_dir / 'metrics'
    run_tb_dir = run_dir / 'tb_logs'
    for directory in (run_dir, run_checkpoint_dir, run_metrics_dir, run_tb_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Publish the active run immediately so dashboards started during training
    # follow the correct isolated directory.
    (DATA_DIR / 'latest_run.txt').write_text(str(run_dir))

    ACTIVE_RUN_CONFIG = {
        'run_tag': run_tag,
        'run_dir': str(run_dir),
        'training_profile': training_profile,
        'observation_mode': observation_mode,
        'line_shape': line_shape,
        'random_trajectories': bool(random_trajectories),
        'curriculum': bool(curriculum),
        'trials_per_trajectory': int(trials_per_trajectory),
        'total_timesteps_requested': int(total_timesteps),
        'n_demos_requested': int(n_demos),
        'seed': int(seed),
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'buffer_size': BUFFER_SIZE,
        'learning_starts': 0 if n_demos > 0 else LEARNING_STARTS,
        'gamma': GAMMA,
        'tau': TAU,
        'train_freq': TRAIN_FREQ,
        'gradient_steps': GRADIENT_STEPS,
        'ent_coef': ent_coef,
        'reward_profile': reward_profile,
        'tripwire_every': int(tripwire_every),
        'tripwire_early_stop': bool(tripwire_early_stop),
    }
    (run_dir / 'config.json').write_text(json.dumps(ACTIVE_RUN_CONFIG, indent=2))

    print(f'[train] run : {run_dir}')
    print(f'[train] profil={training_profile} obs={observation_mode} '
          f'env={"random" if random_trajectories else line_shape}')
    print(f'[train] timesteps={total_timesteps} demos={n_demos} '
          f'curriculum={curriculum} reward={reward_profile} ent_coef={ent_coef}')

    env = make_env(
        line_shape, random_trajectories, headless=headless,
        trials_per_trajectory=trials_per_trajectory, curriculum=curriculum,
        training_profile=training_profile, observation_mode=observation_mode,
        sensor_noise=sensor_noise, reward_profile=reward_profile)

    if run_env_check:
        print('[train] gymnasium check_env...')
        check_env(env.unwrapped, warn=True, skip_render_check=True)

    resume = os.environ.get('RESUME_FROM', '').strip()
    if resume:
        requested = Path(resume).expanduser()
        resume_zip = requested if requested.suffix == '.zip' else Path(str(requested) + '.zip')
        if not resume_zip.exists():
            raise FileNotFoundError(f'[train] RESUME_FROM introuvable : {resume_zip}')
        print(f'[train] reprise : {resume_zip}')
        model = _load_model_checked(str(resume_zip), env)
        model.tensorboard_log = str(run_tb_dir)
        # Preserve the checkpoint entropy mechanism on resume. Reconstructing
        # an automatic-alpha optimizer from a fixed-alpha checkpoint (or the
        # reverse) is unsafe; start a fresh run to change this choice.
        print('[train] reprise : mécanisme entropie du checkpoint conservé')
        for opt in [model.actor.optimizer, model.critic.optimizer]:
            for pg in opt.param_groups:
                pg['lr'] = LEARNING_RATE
        model.replay_buffer.reset()
        print('[train] replay buffer réinitialisé pour le schéma/reward V5')
    else:
        model = SAC(
            'MlpPolicy', env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=(0 if n_demos > 0 else LEARNING_STARTS),
            batch_size=BATCH_SIZE,
            tau=TAU, gamma=GAMMA, train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS, ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=NET_ARCH, activation_fn=torch.nn.ReLU),
            seed=seed, verbose=1, tensorboard_log=str(run_tb_dir),
        )

    demos = []
    if n_demos > 0:
        demo_dir = DATA_DIR / 'demos'
        cache_name = (f'{training_profile}_{observation_mode}_{line_shape}_'
                      f'{reward_profile}_v5.pkl')
        demos = generate_demos(
            env, int(n_demos), save_path=demo_dir / cache_name, seed=seed)
        if demos:
            inject_demos(model, demos, repeat=2)
        else:
            print('[demos] aucune démonstration valide; retour au warm-up aléatoire standard')
            model.learning_starts = LEARNING_STARTS
            ACTIVE_RUN_CONFIG['learning_starts'] = LEARNING_STARTS
            (run_dir / 'config.json').write_text(json.dumps(ACTIVE_RUN_CONFIG, indent=2))

    callbacks = [
        SchemaCheckpointCallback(
            save_freq=SAVE_FREQ, save_path=str(run_checkpoint_dir),
            name_prefix=f'sac_lf_{run_tag}', verbose=1),
        MetricsCallback(log_freq=LOG_FREQ, metrics_dir=run_metrics_dir),
        BestTrainingModelCallback(
            save_dir=run_checkpoint_dir, window=10, min_episodes=5),
        TrajectorySnapshotCallback(
            freq=SNAPSHOT_FREQ, out_dir=run_metrics_dir),
        TripwireCallback(
            every=tripwire_every, early_stop=tripwire_early_stop,
            min_steps=(10_000 if training_profile == 'minimal_straight_line_debug' else 20_000)),
    ]

    t0 = time.time()
    interrupted = False
    try:
        model.learn(
            total_timesteps=total_timesteps, callback=callbacks,
            tb_log_name=f'sac_lf_{training_profile}_{line_shape}',
            reset_num_timesteps=(not bool(resume)))
    except KeyboardInterrupt:
        interrupted = True
        print('[train] interruption utilisateur: sauvegarde de sécurité')
    finally:
        suffix = 'interrupted' if interrupted else 'final'
        final_path = run_checkpoint_dir / f'sac_lf_{line_shape}_{suffix}'
        model.save(str(final_path))
        _save_meta(final_path)
        if save_replay_buffer:
            rb_path = run_checkpoint_dir / 'replay_buffer.pkl'
            model.save_replay_buffer(str(rb_path))
            print(f'[train] replay buffer sauvegardé : {rb_path}')

        # Stable aliases for evaluation scripts while preserving every run.
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        latest_zip = CHECKPOINT_DIR / 'latest_final.zip'
        latest_meta = CHECKPOINT_DIR / 'latest_final.meta.json'
        shutil.copy2(Path(str(final_path) + '.zip'), latest_zip)
        shutil.copy2(final_path.parent / f'{final_path.name}.meta.json', latest_meta)
        (DATA_DIR / 'latest_run.txt').write_text(str(run_dir))
        elapsed = (time.time() - t0) / 3600.0
        print(f'[train] terminé en {elapsed:.2f} h -> {final_path}.zip')
        env.close()


def _reward_profile_from_meta(model_path: str) -> str:
    meta_path = Path(str(model_path).removesuffix('.zip') + '.meta.json')
    if not meta_path.exists():
        return DEFAULT_REWARD_PROFILE
    try:
        profile = json.loads(meta_path.read_text()).get(
            'reward_profile', DEFAULT_REWARD_PROFILE)
    except Exception:
        return DEFAULT_REWARD_PROFILE
    return profile if profile in REWARD_PROFILES else DEFAULT_REWARD_PROFILE


def evaluate(model_path: str, n_episodes: int = 10, line_shape: str = 's_curve',
             training_profile: str = 'realistic',
             observation_mode: str = 'real',
             reward_profile: str | None = None):
    if reward_profile is None:
        reward_profile = _reward_profile_from_meta(model_path)
    env = UR7eLineFollowerEnv(
        line_shape=line_shape, sensor_noise=False, random_trajectories=False,
        observation_mode=observation_mode, training_profile=training_profile,
        curriculum=False, deterministic_pulse=True, reward_profile=reward_profile)
    model = _load_model_checked(model_path, env)
    rewards, successes, progresses, rmses, recent_rmses = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(info.get('is_success', False))
        progresses.append(info.get('progress', 0.0))
        rmses.append(info.get('ep_rmse', np.nan))
        recent_rmses.append(info.get('recent_rmse', np.nan))
        print(
            f"Ep {ep+1:02d} rew={ep_reward:+.2f} prog={progresses[-1]*100:.0f}% "
            f"RMSE={rmses[-1]*100:.1f}cm récente={recent_rmses[-1]*100:.1f}cm "
            f"ok={successes[-1]}")
    print(
        f"Reward moyen={np.mean(rewards):.2f} | succès={np.mean(successes)*100:.0f}% | "
        f"progression={np.mean(progresses)*100:.0f}% | "
        f"RMSE={np.nanmean(rmses)*100:.1f}cm | récente={np.nanmean(recent_rmses)*100:.1f}cm")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='train', choices=['train', 'eval'])
    parser.add_argument('model_path', nargs='?', default='')
    parser.add_argument('n_episodes', nargs='?', type=int, default=10)
    parser.add_argument('--shape', default='s_curve',
                        choices=['s_curve', 'zigzag', 'circle_arc', 'figure_eight',
                                 'horizontal', 'diagonal'])
    parser.add_argument('--profile', default='realistic',
                        choices=['realistic', 'minimal_straight_line_debug'])
    parser.add_argument('--observation-mode', default='real',
                        choices=['real', 'privileged_debug', 'zero'])
    parser.add_argument('--fixed-shape', action='store_true',
                        help='désactive les trajectoires aléatoires')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--demos', type=int, default=N_DEMOS_DEFAULT,
                        help='nombre de démonstrations expertes réussies à injecter')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials-per-drawing', type=int, default=5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--check-env', action='store_true')
    parser.add_argument('--save-replay-buffer', action='store_true')
    parser.add_argument('--reward-profile', choices=REWARD_PROFILES,
                        default=None,
                        help='profil tracking à A/B tester')
    parser.add_argument('--ent-coef', default=str(ENT_COEF_RESET),
                        help="coefficient SAC: 'auto', 'auto_0.05' ou float")
    parser.add_argument('--tripwire-every', type=int, default=1000)
    parser.add_argument('--tripwire-early-stop', action='store_true',
                        help='arrête un run sans tendance après le budget minimal')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.error(
            'aucun argument reçu. Validation recommandée: '
            'ros2 run ur7e_line_follower train --profile minimal_straight_line_debug '
            '--timesteps 50000 --demos 20')

    if args.mode == 'eval':
        path = args.model_path or str(CHECKPOINT_DIR / 'latest_final.zip')
        evaluate(
            path, args.n_episodes, args.shape,
            training_profile=args.profile, observation_mode=args.observation_mode,
            reward_profile=args.reward_profile)
    else:
        train(
            args.shape, random_trajectories=not args.fixed_shape,
            total_timesteps=args.timesteps, headless=not args.visual,
            trials_per_trajectory=args.trials_per_drawing,
            curriculum=not args.no_curriculum,
            training_profile=args.profile,
            observation_mode=args.observation_mode, n_demos=args.demos,
            seed=args.seed, run_env_check=args.check_env,
            save_replay_buffer=args.save_replay_buffer,
            reward_profile=(args.reward_profile or DEFAULT_REWARD_PROFILE), ent_coef=args.ent_coef,
            tripwire_every=args.tripwire_every,
            tripwire_early_stop=args.tripwire_early_stop)


if __name__ == '__main__':
    main()
