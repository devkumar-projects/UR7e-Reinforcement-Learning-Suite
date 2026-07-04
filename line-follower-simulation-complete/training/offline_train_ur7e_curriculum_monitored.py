#!/usr/bin/env python3
"""Offline SAC curriculum/warm-start convergence test for UR7e line follower.

No ROS/Gazebo. Uses the package's analytical FK/MGI, LQR/null-space filter,
EKF, reward and Monte-Carlo line generation. Camera/KLT features are synthetic
privileged features; the real visual pipeline is NOT validated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ur7e_line_follower.control import (
    MAX_WALL_SPEED_M_S,
    wall_action_to_joint_velocity,
)
from ur7e_line_follower.ekf import LaserDotEKF
from ur7e_line_follower.kinematics import (
    fk_ur,
    laser_wall_intersection_unbounded,
)
from ur7e_line_follower.reward import (
    REWARD_PROFILES,
    gated_progress_reward,
    offwall_penalty,
    recent_rmse,
    tracking_reward,
)
from ur7e_line_follower.singularity import (
    lqr_velocity_correction,
    manipulability_obs,
    null_space_manip_correction,
    singularity_penalty,
)
from ur7e_line_follower.target_line import (
    WALL_Y_MAX,
    WALL_Y_MIN,
    WALL_Z_MAX,
    WALL_Z_MIN,
    arc_length,
    closest_point_on_polyline,
    random_line_from_start,
    straight_line_from_start,
    curriculum_line_from_start,
)

HOME_Q = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0], dtype=np.float64)
Q_LOW = np.array([-2*np.pi, -np.pi, 0.0, -2*np.pi, -2*np.pi, -2*np.pi])
Q_HIGH = np.array([2*np.pi, 0.0, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
DT = 0.10
MAX_STEPS = 120
MAX_JVEL = 0.35
PROGRESS_SCALE = MAX_WALL_SPEED_M_S * DT
TRACKING_GATE = 0.10
SUCCESS_RMSE = 0.04
SUCCESS_WINDOW = 30


def point_at_s(line: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(line, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s = float(np.clip(s, 0.0, cum[-1]))
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = min(max(idx, 0), len(seg) - 1)
    length = max(float(lengths[idx]), 1e-9)
    t = (s - cum[idx]) / length
    point = line[idx] + t * seg[idx]
    tangent = seg[idx] / length
    return point.astype(np.float64), tangent.astype(np.float64)


class OfflineUR7eEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_profile: str, random_lines: bool = False, seed: int = 0, line_level: int = -1):
        super().__init__()
        self.reward_profile = reward_profile
        self.random_lines = bool(random_lines)
        self.line_level = int(line_level)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(33,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.ekf = LaserDotEKF(dt=DT)
        self.recent_errors = deque(maxlen=SUCCESS_WINDOW)
        self.reset(seed=seed)

    def _dot(self) -> np.ndarray | None:
        return laser_wall_intersection_unbounded(self.q)

    @staticmethod
    def _on_wall(dot: np.ndarray | None) -> bool:
        return bool(
            dot is not None
            and WALL_Y_MIN <= dot[0] <= WALL_Y_MAX
            and WALL_Z_MIN <= dot[1] <= WALL_Z_MAX
        )

    def _geometry(self):
        dot = self._dot()
        used = self.last_valid_dot.copy() if dot is None else np.asarray(dot, dtype=np.float64)
        if dot is not None:
            self.last_valid_dot = used.copy()
        closest = closest_point_on_polyline(used, self.line)
        dist = float(closest["distance"])
        s = float(closest["abscissa"])
        target, tangent = point_at_s(self.line, min(s + 0.04, self.path_length))
        return dot, used, closest, dist, s, target, tangent

    def _obs(self) -> np.ndarray:
        dot, used, closest, dist, s, target, tangent = self._geometry()
        on_wall = self._on_wall(dot)
        progress = float(np.clip(s / max(self.path_length, 1e-9), 0.0, 1.0))
        q_norm = 2.0 * (self.q - Q_LOW) / np.maximum(Q_HIGH - Q_LOW, 1e-9) - 1.0
        tcp = fk_ur(self.q)
        guidance = np.array([
            np.clip((target[0] - used[0]) / 0.10, -1.0, 1.0),
            np.clip((target[1] - used[1]) / 0.10, -1.0, 1.0),
            progress,
        ])
        # Synthetic camera/KLT-like features for the offline surrogate only.
        normal = np.array([-tangent[1], tangent[0]])
        signed_offset = float(np.dot(used - closest["closest"], normal))
        klt_conf = float(np.exp(-dist / 0.08)) if on_wall else 0.0
        cam = np.array([
            float(on_wall),
            np.clip(signed_offset / 0.20, -1.0, 1.0),
            klt_conf,
            np.clip(tangent[0], -1.0, 1.0),
            np.clip(tangent[1], -1.0, 1.0),
            progress,
            float(on_wall),
        ])
        ekf_pos = self.ekf.position
        ekf_sigma = np.clip(self.ekf.uncertainty / 0.05, 0.0, 2.0)
        manip = manipulability_obs(self.q)
        obs = np.concatenate([
            q_norm, tcp, used, [float(on_wall)], guidance, cam,
            ekf_pos, ekf_sigma, manip,
            self.previous_action,
            self.previous_wall_velocity / MAX_WALL_SPEED_M_S,
        ]).astype(np.float32)
        return np.clip(np.nan_to_num(obs), -2.0, 2.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q = HOME_Q.copy()
        initial_dot = laser_wall_intersection_unbounded(self.q)
        if initial_dot is None:
            raise RuntimeError("HOME_Q ne projette pas le laser sur le mur")
        self.last_valid_dot = np.asarray(initial_dot, dtype=np.float64)
        if self.line_level >= 0:
            self.line = curriculum_line_from_start(
                self.rng, self.last_valid_dot, level=self.line_level
            )
        elif self.random_lines:
            self.line = random_line_from_start(self.rng, self.last_valid_dot)
        else:
            self.line = straight_line_from_start(self.last_valid_dot, length=0.25)
        self.path_length = arc_length(self.line)
        self.step_count = 0
        self.offwall_steps = 0
        self.previous_s = 0.0
        self.max_s = 0.0
        self.previous_action = np.zeros(2, dtype=np.float64)
        self.previous_wall_velocity = np.zeros(2, dtype=np.float64)
        self.recent_errors.clear()
        self.ekf.reset(float(initial_dot[0]), float(initial_dot[1]))
        return self._obs(), {}

    def set_curriculum_level(self, level: int):
        self.line_level = int(level)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        qdot_raw, wall_velocity = wall_action_to_joint_velocity(
            self.q, action, self.previous_wall_velocity,
        )
        qdot_lqr = lqr_velocity_correction(
            self.q, qdot_raw, q_dot_max=MAX_JVEL,
        )
        qdot_null = null_space_manip_correction(
            self.q, alpha=0.03, max_norm=0.05,
        )
        qdot = np.clip(qdot_lqr + qdot_null, -MAX_JVEL, MAX_JVEL)
        self.q = np.clip(self.q + qdot * DT, Q_LOW, Q_HIGH)

        dot, used, closest, dist, s, _, _ = self._geometry()
        on_wall = self._on_wall(dot)
        self.ekf.predict(self.q, qdot, dt=DT)
        if dot is not None:
            measurement = used + self.rng.normal(0.0, 0.003, 2)
            self.ekf.update_fk(float(measurement[0]), float(measurement[1]))

        delta_s = s - self.previous_s
        r_track = tracking_reward(dist, profile=self.reward_profile)
        r_progress = gated_progress_reward(
            delta_s, dist,
            nominal_step_m=PROGRESS_SCALE,
            gain=2.0,
            tracking_gate_m=TRACKING_GATE,
        )
        r_smooth = -0.02 * float(np.sum((action - self.previous_action) ** 2))
        r_action = -0.01 * float(np.sum(action ** 2))
        r_sing = singularity_penalty(self.q)
        r_offwall = 0.0
        if on_wall:
            self.offwall_steps = 0
        else:
            self.offwall_steps += 1
            dy = max(WALL_Y_MIN - used[0], 0.0, used[0] - WALL_Y_MAX)
            dz = max(WALL_Z_MIN - used[1], 0.0, used[1] - WALL_Z_MAX)
            r_offwall = offwall_penalty(float(np.hypot(dy, dz)))

        if dist <= TRACKING_GATE:
            self.max_s = max(self.max_s, s)
        self.recent_errors.append(dist)
        coverage = float(np.clip(self.max_s / max(self.path_length, 1e-9), 0.0, 1.0))
        rmse_recent = recent_rmse(list(self.recent_errors), SUCCESS_WINDOW)
        success = bool(coverage >= 0.985 and rmse_recent <= SUCCESS_RMSE and on_wall)

        reward = 0.1 * (r_track + r_progress + r_smooth + r_action + r_sing + r_offwall)
        if success:
            reward += 10.0
        if self.offwall_steps >= 5:
            reward -= 3.0

        self.step_count += 1
        terminated = success or self.offwall_steps >= 5
        truncated = self.step_count >= MAX_STEPS
        self.previous_s = s
        self.previous_action = action.copy()
        self.previous_wall_velocity = wall_velocity.copy()

        info = {
            "is_success": success,
            "progress": coverage,
            "recent_rmse": rmse_recent,
            "distance": dist,
            "on_wall": on_wall,
            "reward_tracking": r_track,
            "reward_progress": r_progress,
        }
        return self._obs(), float(reward), terminated, truncated, info



LEVEL_SPECS = [
    ("fixed_line", False, -1),
    ("curriculum_level_0_gentle", False, 0),
    ("curriculum_level_1_moderate", False, 1),
    ("monte_carlo_random_lines", True, 2),
]


def _safe_mean(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_detailed(model: SAC, reward_profile: str, episodes: int,
                      random_lines: bool, seed: int, line_level: int = -1,
                      capture_trace: bool = False):
    env = OfflineUR7eEnv(
        reward_profile,
        random_lines=random_lines,
        seed=seed,
        line_level=line_level,
    )
    successes, progresses, rmses, returns, lengths = [], [], [], [], []
    trace = None
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        info = {}
        dots = []
        if capture_trace and ep == 0:
            dot = env._dot()
            if dot is not None:
                dots.append(np.asarray(dot, dtype=float).copy())
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if capture_trace and ep == 0:
                dot = env._dot()
                if dot is not None:
                    dots.append(np.asarray(dot, dtype=float).copy())
        successes.append(float(info.get("is_success", False)))
        progresses.append(float(info.get("progress", 0.0)))
        rmses.append(float(info.get("recent_rmse", np.nan)))
        returns.append(total)
        lengths.append(steps)
        if capture_trace and ep == 0:
            trace = {
                "target": np.asarray(env.line, dtype=float),
                "laser": np.asarray(dots, dtype=float),
                "success": bool(info.get("is_success", False)),
                "progress": float(info.get("progress", 0.0)),
                "recent_rmse_m": float(info.get("recent_rmse", np.nan)),
                "return": float(total),
                "steps": int(steps),
            }
    metrics = {
        "episodes": int(episodes),
        "success_rate": float(np.mean(successes)),
        "mean_progress": float(np.mean(progresses)),
        "mean_recent_rmse_m": float(np.nanmean(rmses)),
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }
    return metrics, trace


def evaluate_suite(model: SAC, reward_profile: str, episodes: int, seed: int,
                   capture_traces: bool = False):
    result = {}
    traces = {}
    for idx, (name, random_lines, line_level) in enumerate(LEVEL_SPECS):
        metrics, trace = evaluate_detailed(
            model,
            reward_profile,
            episodes,
            random_lines,
            seed + idx * 10_000,
            line_level=line_level,
            capture_trace=capture_traces,
        )
        result[name] = metrics
        if trace is not None:
            traces[name] = trace
    return result, traces


def save_trajectory_plot(trace: dict, path: Path, title: str):
    target = np.asarray(trace.get("target", []), dtype=float)
    laser = np.asarray(trace.get("laser", []), dtype=float)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    if target.ndim == 2 and len(target):
        ax.plot(target[:, 0], target[:, 1], linewidth=2.2, label="Cible")
    if laser.ndim == 2 and len(laser):
        ax.plot(laser[:, 0], laser[:, 1], linewidth=1.5, label="Laser SAC")
    ax.set_xlabel("y mur [m]")
    ax.set_ylabel("z mur [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_series(x, ys, labels, ylabel, title, path: Path):
    if not x:
        return
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    for y, label in zip(ys, labels):
        ax.plot(x, y, marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(labels) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_training_plots(run_dir: Path):
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(run_dir / "episode_metrics.csv")
    if rows:
        steps = [int(float(r["timestep"])) for r in rows]
        reward = [float(r["episode_return"]) for r in rows]
        progress = [100.0 * float(r["progress"]) for r in rows]
        success = [100.0 * float(r["success"]) for r in rows]
        rmse = [100.0 * float(r["recent_rmse_m"]) for r in rows]

        def rolling(vals, window=50):
            out = []
            for i in range(len(vals)):
                chunk = vals[max(0, i-window+1):i+1]
                out.append(float(np.nanmean(chunk)))
            return out

        _plot_series(steps, [rolling(reward)], ["Reward moyen 50 ep"],
                     "Reward", "Entraînement — reward", plot_dir / "training_reward.png")
        _plot_series(steps, [rolling(progress)], ["Progression moyenne 50 ep"],
                     "Progression [%]", "Entraînement — progression", plot_dir / "training_progress.png")
        _plot_series(steps, [rolling(success)], ["Succès moyen 50 ep"],
                     "Succès [%]", "Entraînement — taux de succès", plot_dir / "training_success.png")
        _plot_series(steps, [rolling(rmse)], ["RMSE moyen 50 ep"],
                     "RMSE [cm]", "Entraînement — RMSE récente", plot_dir / "training_rmse.png")

    eval_rows = _read_csv(run_dir / "evaluation_history.csv")
    if eval_rows:
        steps = [int(float(r["timestep"])) for r in eval_rows]
        labels = [name for name, _, _ in LEVEL_SPECS]
        success_ys = [[100.0 * float(r[f"{name}_success_rate"]) for r in eval_rows] for name in labels]
        progress_ys = [[100.0 * float(r[f"{name}_mean_progress"]) for r in eval_rows] for name in labels]
        rmse_ys = [[100.0 * float(r[f"{name}_mean_recent_rmse_m"]) for r in eval_rows] for name in labels]
        return_ys = [[float(r[f"{name}_mean_return"]) for r in eval_rows] for name in labels]
        _plot_series(steps, success_ys, labels, "Succès [%]",
                     "Évaluations globales — succès", plot_dir / "evaluation_success.png")
        _plot_series(steps, progress_ys, labels, "Progression [%]",
                     "Évaluations globales — progression", plot_dir / "evaluation_progress.png")
        _plot_series(steps, rmse_ys, labels, "RMSE [cm]",
                     "Évaluations globales — RMSE", plot_dir / "evaluation_rmse.png")
        _plot_series(steps, return_ys, labels, "Return",
                     "Évaluations globales — return", plot_dir / "evaluation_return.png")


class OfflineMonitoringCallback(BaseCallback):
    def __init__(self, run_dir: Path, reward_profile: str, seed: int,
                 every: int = 2000, checkpoint_freq: int = 10_000,
                 eval_freq: int = 10_000, eval_episodes: int = 20,
                 curriculum_schedule=None, initial_level: int | None = None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.reward_profile = reward_profile
        self.seed = int(seed)
        self.every = max(1, int(every))
        self.checkpoint_freq = max(0, int(checkpoint_freq))
        self.eval_freq = max(0, int(eval_freq))
        self.eval_episodes = max(1, int(eval_episodes))
        self.curriculum_schedule = curriculum_schedule or []
        self.current_level = initial_level
        self.episode_index = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.successes = deque(maxlen=50)
        self.progresses = deque(maxlen=50)
        self.rmses = deque(maxlen=50)
        self.returns = deque(maxlen=50)
        self.lengths = deque(maxlen=50)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.snapshot_dir = self.run_dir / "evaluation_snapshots"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.episode_csv = self.run_dir / "episode_metrics.csv"
        self.eval_csv = self.run_dir / "evaluation_history.csv"
        self.summary_json = self.run_dir / "training_summary.json"
        self.latest_eval_json = self.run_dir / "evaluation_latest.json"
        self._prepare_files()

    def _prepare_files(self):
        if not self.episode_csv.exists():
            with self.episode_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestep", "episode", "curriculum_level", "episode_return",
                    "episode_length", "success", "progress", "recent_rmse_m",
                    "distance_m", "on_wall",
                ])
        if not self.eval_csv.exists():
            fields = ["timestep"]
            for name, _, _ in LEVEL_SPECS:
                fields.extend([
                    f"{name}_success_rate",
                    f"{name}_mean_progress",
                    f"{name}_mean_recent_rmse_m",
                    f"{name}_mean_return",
                    f"{name}_mean_episode_length",
                ])
            with self.eval_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

    def _set_curriculum(self):
        if not self.curriculum_schedule:
            return
        level = self.curriculum_schedule[0][1]
        for threshold, candidate in self.curriculum_schedule:
            if self.num_timesteps >= threshold:
                level = candidate
        if level != self.current_level:
            self.training_env.env_method("set_curriculum_level", int(level))
            self.current_level = int(level)
            print(f"[offline] curriculum -> level {self.current_level} @ step {self.num_timesteps}")

    def _write_episode(self, info: dict):
        success = float(info.get("is_success", False))
        progress = float(info.get("progress", 0.0))
        rmse = float(info.get("recent_rmse", np.nan))
        distance = float(info.get("distance", np.nan))
        on_wall = float(info.get("on_wall", False))
        self.successes.append(success)
        self.progresses.append(progress)
        self.rmses.append(rmse)
        self.returns.append(self.episode_return)
        self.lengths.append(self.episode_length)
        self.episode_index += 1
        with self.episode_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                self.num_timesteps,
                self.episode_index,
                "fixed" if self.current_level is None else self.current_level,
                f"{self.episode_return:.9g}",
                self.episode_length,
                f"{success:.0f}",
                f"{progress:.9g}",
                f"{rmse:.9g}",
                f"{distance:.9g}",
                f"{on_wall:.0f}",
            ])
        self.episode_return = 0.0
        self.episode_length = 0

    def _summary(self):
        summary = {
            "timestep": int(self.num_timesteps),
            "episodes": int(self.episode_index),
            "curriculum_level": self.current_level,
            "rolling_50": {
                "success_rate": _safe_mean(self.successes),
                "mean_progress": _safe_mean(self.progresses),
                "mean_recent_rmse_m": _safe_mean(self.rmses),
                "mean_return": _safe_mean(self.returns),
                "mean_episode_length": _safe_mean(self.lengths),
            },
            "device": str(self.model.device),
            "last_update_unix": time.time(),
        }
        self.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.logger.record("offline/success_rate_50", summary["rolling_50"]["success_rate"])
        self.logger.record("offline/progress_50", summary["rolling_50"]["mean_progress"])
        self.logger.record("offline/rmse_50_m", summary["rolling_50"]["mean_recent_rmse_m"])
        if self.current_level is not None:
            self.logger.record("offline/curriculum_level", self.current_level)
        return summary

    def _save_checkpoint(self):
        path = self.checkpoint_dir / f"offline_sac_step_{self.num_timesteps:09d}"
        self.model.save(str(path))
        meta = self._summary()
        (path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[offline] checkpoint={path}.zip")

    def _run_evaluation(self):
        print(f"[offline] évaluation globale @ step {self.num_timesteps}...")
        result, traces = evaluate_suite(
            self.model,
            self.reward_profile,
            self.eval_episodes,
            self.seed + 100_000 + self.num_timesteps,
            capture_traces=True,
        )
        payload = {"timestep": int(self.num_timesteps), "results": result}
        snap_json = self.snapshot_dir / f"evaluation_step_{self.num_timesteps:09d}.json"
        snap_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.latest_eval_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        row = [self.num_timesteps]
        for name, _, _ in LEVEL_SPECS:
            metrics = result[name]
            row.extend([
                metrics["success_rate"], metrics["mean_progress"],
                metrics["mean_recent_rmse_m"], metrics["mean_return"],
                metrics["mean_episode_length"],
            ])
        with self.eval_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        for name, trace in traces.items():
            save_trajectory_plot(
                trace,
                self.snapshot_dir / f"trajectory_{name}_step_{self.num_timesteps:09d}.png",
                f"{name} — step {self.num_timesteps:,} — progress {trace['progress']*100:.1f}%",
            )
        generate_training_plots(self.run_dir)
        compact = " | ".join(
            f"{name}: S={metrics['success_rate']*100:.0f}% P={metrics['mean_progress']*100:.0f}% RMSE={metrics['mean_recent_rmse_m']*100:.1f}cm"
            for name, metrics in result.items()
        )
        print(f"[offline] eval {compact}")

    def _on_step(self) -> bool:
        self._set_curriculum()
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        if len(rewards):
            self.episode_return += float(rewards[0])
            self.episode_length += 1
        for done, info in zip(dones, infos):
            if done:
                self._write_episode(info)

        if self.num_timesteps % self.every == 0:
            summary = self._summary()
            r = summary["rolling_50"]
            level = "fixed" if self.current_level is None else str(self.current_level)
            print(
                f"[offline] step={self.num_timesteps:7d} level={level} "
                f"success50={100*r['success_rate']:5.1f}% "
                f"progress50={100*r['mean_progress']:5.1f}% "
                f"rmse50={100*r['mean_recent_rmse_m']:5.2f}cm "
                f"return50={r['mean_return']:+7.2f}"
            )

        if self.checkpoint_freq and self.num_timesteps % self.checkpoint_freq == 0:
            self._save_checkpoint()
        if self.eval_freq and self.num_timesteps % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _on_training_end(self):
        self._summary()
        generate_training_plots(self.run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="normalized_huber")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--net", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--random-lines", action="store_true")
    parser.add_argument("--line-level", type=int, default=-1, choices=[-1, 0, 1, 2],
                        help="-1=fixed line, 0=gentle curves, 1=moderate curves, 2=full random")
    parser.add_argument("--load-model", default=None,
                        help="Warm-start from an existing SB3 SAC .zip model")
    parser.add_argument("--load-replay-buffer", default=None,
                        help="Optional replay buffer .pkl to restore after loading the model")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use staged levels 0 -> 1 -> 2 during this run")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Final evaluation episodes per level")
    parser.add_argument("--periodic-eval-episodes", type=int, default=20,
                        help="Episodes per level for periodic snapshots")
    parser.add_argument("--report-freq", type=int, default=2_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or stamp
    run_dir = Path.home() / ".ros" / "ur7e_line_follower" / "offline_runs" / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    initial_level = 0 if args.curriculum else args.line_level
    env = Monitor(
        OfflineUR7eEnv(args.reward_profile, args.random_lines, args.seed, line_level=initial_level),
        filename=str(run_dir / "monitor.csv"),
        info_keywords=("is_success", "progress", "recent_rmse", "distance", "on_wall"),
    )

    if args.load_model:
        model_path = str(Path(args.load_model).expanduser().resolve())
        print(f"[offline] warm-start model={model_path}")
        model = SAC.load(model_path, env=env, device=args.device)
        model.tensorboard_log = str(run_dir / "tb")
        model.verbose = 1
        if args.load_replay_buffer:
            replay_path = str(Path(args.load_replay_buffer).expanduser().resolve())
            model.load_replay_buffer(replay_path)
            print(f"[offline] replay buffer chargé={replay_path}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=args.buffer_size,
            learning_starts=1_000,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=args.net, activation_fn=torch.nn.ReLU),
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
        )

    print(f"[offline] run={run_dir}")
    print(f"[offline] device demandé={args.device}, device SB3={model.device}")
    schedule = []
    if args.curriculum:
        schedule = [
            (0, 0),
            (max(1, args.timesteps // 3), 1),
            (max(2, 2 * args.timesteps // 3), 2),
        ]

    callback = OfflineMonitoringCallback(
        run_dir=run_dir,
        reward_profile=args.reward_profile,
        seed=args.seed,
        every=args.report_freq,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.periodic_eval_episodes,
        curriculum_schedule=schedule,
        initial_level=initial_level,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="SAC",
    )
    model.save(str(run_dir / "offline_sac_final"))
    if args.save_replay_buffer:
        model.save_replay_buffer(str(run_dir / "offline_replay_buffer.pkl"))
        print(f"[offline] replay buffer final={run_dir / 'offline_replay_buffer.pkl'}")

    result, traces = evaluate_suite(
        model,
        args.reward_profile,
        args.eval_episodes,
        args.seed + 500_000,
        capture_traces=True,
    )
    final_payload = {"timestep": int(args.timesteps), "results": result}
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "evaluation_final_detailed.json").write_text(
        json.dumps(final_payload, indent=2), encoding="utf-8"
    )
    for name, trace in traces.items():
        save_trajectory_plot(
            trace,
            run_dir / "evaluation_snapshots" / f"trajectory_{name}_final.png",
            f"{name} — final — progress {trace['progress']*100:.1f}%",
        )
    generate_training_plots(run_dir)

    print("\n========== ÉVALUATION OFFLINE FINALE ==========")
    print(json.dumps(result, indent=2))
    print(f"Modèle, checkpoints, courbes et résultats : {run_dir}")
    if result["fixed_line"]["success_rate"] < 0.80:
        raise SystemExit("NO-GO: le modèle ne conserve pas la convergence sur la droite fixe")


if __name__ == "__main__":
    main()
