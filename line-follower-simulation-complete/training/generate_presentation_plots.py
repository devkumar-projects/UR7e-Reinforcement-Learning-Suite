#!/usr/bin/env python3
"""
UR7e Line Follower — Génération de courbes pour présentation.
10 épisodes de test avec le modèle SAC entraîné.
Génère des PNG pour : KLT, MGD/MGI, EKF/Kalman, Monte Carlo, LQR.
Usage :
    cd ~/ur7e_training/ur7e_line_follower
    PYTHONPATH="$PWD" python generate_presentation_plots.py \
        --model ~/.ros/ur7e_line_follower/offline_runs/basile_level2/offline_sac_final.zip \
        --outdir ~/presentation_plots
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from stable_baselines3 import SAC

from offline_train_ur7e_curriculum_monitored import OfflineUR7eEnv, HOME_Q, DT

from ur7e_line_follower.singularity import (
    lqr_velocity_correction,
    null_space_manip_correction,
    manipulability_obs,
    command_filter_diagnostics,
)
from ur7e_line_follower.kinematics import fk_ur, wall_jacobian
from ur7e_line_follower.ekf import LaserDotEKF

# ── Palette présentation ──────────────────────────────────────────────────────
C = {
    "klt_off":   "#2196F3",
    "klt_conf":  "#4CAF50",
    "klt_tan":   "#FF9800",
    "klt_cov":   "#9C27B0",
    "fk_y":      "#1565C0",
    "fk_z":      "#AD1457",
    "target_y":  "#42A5F5",
    "target_z":  "#EC407A",
    "error":     "#E53935",
    "ekf_y":     "#00796B",
    "ekf_z":     "#BF360C",
    "sigma_y":   "#80CBC4",
    "sigma_z":   "#FFAB91",
    "noise":     "#B0BEC5",
    "lqr_vy":    "#1B5E20",
    "lqr_vz":    "#F57F17",
    "raw_vy":    "#A5D6A7",
    "raw_vz":    "#FFF9C4",
    "manip":     "#6A1B9A",
    "success":   "#00C853",
    "failure":   "#D50000",
    "line_traj": "#546E7A",
    "dot_traj":  "#E91E63",
}

FIGSIZE_WIDE = (16, 9)
FIGSIZE_TALL = (12, 14)
DPI = 150


# ── Enregistrement d'un épisode ───────────────────────────────────────────────
def run_episode(model, env: OfflineUR7eEnv, seed: int, line_level: int):
    env.line_level = line_level
    obs, _ = env.reset(seed=seed)

    rec = dict(
        # temps
        t=[],
        # KLT synthétique (obs[15:22])
        klt_detected=[], klt_offset=[], klt_conf=[], klt_tan_cos=[], klt_tan_sin=[],
        klt_coverage=[], klt_laser=[],
        # MGD (obs[6:9] = tcp,  obs[9:11] = dot_yz)
        tcp_x=[], tcp_y=[], tcp_z=[],
        dot_y=[], dot_z=[], on_wall=[],
        # cible
        target_y=[], target_z=[], track_err=[],
        # EKF (obs[22:24] position, obs[24:26] sigma)
        ekf_y=[], ekf_z=[], ekf_sig_y=[], ekf_sig_z=[],
        # EKF brut (depuis l'objet)
        ekf_raw_y=[], ekf_raw_z=[], ekf_cov_yy=[], ekf_cov_zz=[],
        # LQR (actions, vitesses mur, manipulabilité)
        act_vy=[], act_vz=[],
        wall_vy=[], wall_vz=[],
        manip_w=[],
        # récompense
        reward=[], r_track=[], r_prog=[],
        # progression
        progress=[], rmse=[],
        # joints
        q=[],
    )

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # — indices dans obs (33D schema V5) —
        # [0:6]  q_norm
        # [6:9]  tcp
        # [9:11] dot_yz
        # [11]   on_wall
        # [12:15] visual_guidance (target_rel_y, target_rel_z, progress)
        # [15:22] camera_v4 (detected, offset, conf, tan_cos, tan_sin, cov, laser)
        # [22:24] ekf_pos
        # [24:26] ekf_sigma
        # [26:29] manip
        # [29:31] prev_action
        # [31:33] prev_wall_vel

        rec["t"].append(step * DT)
        rec["klt_detected"].append(float(obs[15]))
        rec["klt_offset"].append(float(obs[16]))
        rec["klt_conf"].append(float(obs[17]))
        rec["klt_tan_cos"].append(float(obs[18]))
        rec["klt_tan_sin"].append(float(obs[19]))
        rec["klt_coverage"].append(float(obs[20]))
        rec["klt_laser"].append(float(obs[21]))
        rec["tcp_x"].append(float(obs[6]))
        rec["tcp_y"].append(float(obs[7]))
        rec["tcp_z"].append(float(obs[8]))
        rec["dot_y"].append(float(obs[9]))
        rec["dot_z"].append(float(obs[10]))
        rec["on_wall"].append(float(obs[11]))
        rec["ekf_y"].append(float(obs[22]))
        rec["ekf_z"].append(float(obs[23]))
        rec["ekf_sig_y"].append(float(obs[24]))
        rec["ekf_sig_z"].append(float(obs[25]))
        rec["ekf_raw_y"].append(float(env.ekf.position[0]))
        rec["ekf_raw_z"].append(float(env.ekf.position[1]))
        cov = env.ekf._P
        rec["ekf_cov_yy"].append(float(abs(cov[0, 0])))
        rec["ekf_cov_zz"].append(float(abs(cov[1, 1])))
        rec["manip_w"].append(float(obs[26]))
        rec["act_vy"].append(float(action[0]))
        rec["act_vz"].append(float(action[1]))
        rec["wall_vy"].append(float(env.previous_wall_velocity[0]))
        rec["wall_vz"].append(float(env.previous_wall_velocity[1]))
        rec["reward"].append(float(reward))
        rec["r_track"].append(float(info.get("reward_tracking", 0.0)))
        rec["r_prog"].append(float(info.get("reward_progress", 0.0)))
        rec["progress"].append(float(info.get("progress", 0.0)))
        rec["rmse"].append(float(info.get("recent_rmse", 0.0)))
        rec["track_err"].append(float(info.get("distance", 0.0)))
        rec["q"].append(env.q.copy())

        # position cible (approx via guidance)
        dot_y, dot_z = float(obs[9]), float(obs[10])
        g_dy = float(obs[12]) * 0.10
        g_dz = float(obs[13]) * 0.10
        rec["target_y"].append(dot_y + g_dy)
        rec["target_z"].append(dot_z + g_dz)

        total_reward += reward
        obs = next_obs
        step += 1

    for k in rec:
        if isinstance(rec[k], list) and rec[k] and isinstance(rec[k][0], np.ndarray):
            rec[k] = np.array(rec[k])
        else:
            rec[k] = np.array(rec[k])

    rec["success"] = bool(info.get("is_success", False))
    rec["total_reward"] = total_reward
    rec["final_progress"] = float(info.get("progress", 0.0))
    rec["line"] = env.line.copy()
    rec["line_level"] = line_level
    return rec


# ── Plot 1 : KLT — 10 épisodes superposés ────────────────────────────────────
def plot_klt(episodes: list[dict], outdir: Path):
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_WIDE, dpi=DPI, sharex=False)
    fig.suptitle("KLT Tracking — Lucas-Kanade Optical Flow\n(10 episodes)", fontsize=16, fontweight="bold")

    ax_off, ax_conf, ax_tan = axes

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, ep in enumerate(episodes):
        t = ep["t"]
        label = f"Ep{i+1} {'✓' if ep['success'] else '✗'}"
        ax_off.plot(t, ep["klt_offset"], color=colors[i], alpha=0.8, lw=1.5, label=label)
        ax_conf.plot(t, ep["klt_conf"], color=colors[i], alpha=0.8, lw=1.5)
        ax_tan.plot(t, np.degrees(np.arctan2(ep["klt_tan_sin"], ep["klt_tan_cos"])),
                    color=colors[i], alpha=0.8, lw=1.5)

    ax_off.axhline(0, color="k", lw=0.8, ls="--")
    ax_off.axhline(0.2, color="gray", lw=0.6, ls=":")
    ax_off.axhline(-0.2, color="gray", lw=0.6, ls=":")
    ax_off.fill_between([0, 15], -0.2, 0.2, alpha=0.07, color="green", label="Zone tracking OK")
    ax_off.set_ylabel("Normalized lateral offset [-1, 1]", fontsize=11)
    ax_off.set_title("KLT lateral offset (error relative to the line)", fontsize=12)
    ax_off.legend(fontsize=8, ncol=5, loc="upper right")
    ax_off.grid(True, alpha=0.3)
    ax_off.set_ylim(-1.1, 1.1)

    ax_conf.axhline(0.5, color="orange", lw=0.8, ls="--", label="Seuil confiance 0.5")
    ax_conf.set_ylabel("Confiance KLT [0, 1]", fontsize=11)
    ax_conf.set_title("KLT tracker confidence (stable points ratio x retention)", fontsize=12)
    ax_conf.legend(fontsize=9)
    ax_conf.grid(True, alpha=0.3)
    ax_conf.set_ylim(-0.05, 1.1)

    ax_tan.axhline(0, color="k", lw=0.8, ls="--")
    ax_tan.set_ylabel("Tangent angle (deg)", fontsize=11)
    ax_tan.set_xlabel("Time (s)", fontsize=11)
    ax_tan.set_title("Line tangent orientation (EMA alpha=0.30)", fontsize=12)
    ax_tan.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "01_klt_tracking.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Plot 2 : MGD/MGI — trajectoires sur le mur ───────────────────────────────
def plot_kinematics(episodes: list[dict], outdir: Path):
    fig = plt.figure(figsize=FIGSIZE_WIDE, dpi=DPI)
    fig.suptitle("FK / IK UR7e — Trajectories on the wall\n(10 episodes)", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_wall = fig.add_subplot(gs[:, 0:2])   # trajectoires mur (grand)
    ax_erry = fig.add_subplot(gs[0, 2])     # erreur Y
    ax_errz = fig.add_subplot(gs[1, 2])     # erreur Z

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    from ur7e_line_follower.target_line import WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX

    # Mur
    wall_rect = plt.Rectangle((WALL_Y_MIN, WALL_Z_MIN),
                               WALL_Y_MAX - WALL_Y_MIN, WALL_Z_MAX - WALL_Z_MIN,
                               fill=True, facecolor="#ECEFF1", edgecolor="#90A4AE", lw=2)
    ax_wall.add_patch(wall_rect)
    ax_wall.set_xlim(WALL_Y_MIN - 0.05, WALL_Y_MAX + 0.05)
    ax_wall.set_ylim(WALL_Z_MIN - 0.05, WALL_Z_MAX + 0.05)
    ax_wall.set_aspect("equal")
    ax_wall.set_xlabel("Wall Y (m)", fontsize=11)
    ax_wall.set_ylabel("Z mur (m)", fontsize=11)
    ax_wall.set_title("Laser trajectories on the wall\n(target line vs actual tracking)", fontsize=12)
    ax_wall.grid(True, alpha=0.25)

    for i, ep in enumerate(episodes):
        line = ep["line"]
        # Ligne cible
        ax_wall.plot(line[:, 0], line[:, 1], "--", color=colors[i], lw=1.2, alpha=0.5)
        # Actual trajectory du laser
        ax_wall.plot(ep["dot_y"], ep["dot_z"], "-", color=colors[i], lw=2.0, alpha=0.85,
                     label=f"Ep{i+1} {'✓' if ep['success'] else '✗'}")
        ax_wall.plot(ep["dot_y"][0], ep["dot_z"][0], "o", color=colors[i], ms=6)

    ax_wall.legend(fontsize=8, ncol=2, loc="lower right")

    legend_elements = [
        Line2D([0], [0], ls="--", color="gray", lw=1.5, label="Ligne cible"),
        Line2D([0], [0], ls="-", color="gray", lw=2, label="Actual trajectory"),
    ]
    ax_wall.legend(handles=legend_elements + [
        Line2D([0], [0], color=colors[i], lw=2,
               label=f"Ep{i+1} {'✓' if ep['success'] else '✗'}")
        for i, ep in enumerate(episodes)
    ], fontsize=7.5, ncol=2, loc="lower right")

    for i, ep in enumerate(episodes):
        t = ep["t"]
        err_y = np.array(ep["dot_y"]) - np.array(ep["target_y"])
        err_z = np.array(ep["dot_z"]) - np.array(ep["target_z"])
        ax_erry.plot(t, err_y * 100, color=colors[i], lw=1.5, alpha=0.8)
        ax_errz.plot(t, err_z * 100, color=colors[i], lw=1.5, alpha=0.8)

    for ax, label in [(ax_erry, "Erreur Y (cm)"), (ax_errz, "Erreur Z (cm)")]:
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axhline(4, color="orange", lw=0.8, ls=":", label="+/-4 cm")
        ax.axhline(-4, color="orange", lw=0.8, ls=":")
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_erry.set_title("Erreur MGI : axe Y", fontsize=11)
    ax_errz.set_title("Erreur MGI : axe Z", fontsize=11)

    plt.tight_layout()
    out = outdir / "02_mgd_mgi_trajectoires.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Plot 3 : EKF / Kalman ─────────────────────────────────────────────────────
def plot_ekf(episodes: list[dict], outdir: Path):
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE_TALL, dpi=DPI)
    fig.suptitle("Extended Kalman Filter (EKF)\nEstimating the laser dot position on the wall",
                 fontsize=16, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Paramètres EKF du système
    q_pos_std = 0.002
    q_vel_std = 0.10
    r_fk_std = 0.006
    r_cam_std = 0.018

    for i, ep in enumerate(episodes):
        t = ep["t"]
        c = colors[i]
        alpha = 0.75

        # Position Y estimée vs mesure FK
        axes[0, 0].plot(t, ep["ekf_raw_y"], color=c, lw=1.5, alpha=alpha)
        axes[0, 1].plot(t, ep["ekf_raw_z"], color=c, lw=1.5, alpha=alpha)

        # Variance (diagonale P)
        sigma_y_mm = np.sqrt(np.clip(ep["ekf_cov_yy"], 0, None)) * 1000
        sigma_z_mm = np.sqrt(np.clip(ep["ekf_cov_zz"], 0, None)) * 1000
        axes[1, 0].plot(t, sigma_y_mm, color=c, lw=1.5, alpha=alpha)
        axes[1, 1].plot(t, sigma_z_mm, color=c, lw=1.5, alpha=alpha)

        # Erreur EKF vs vraie position
        err_y = (ep["ekf_raw_y"] - ep["dot_y"]) * 1000
        err_z = (ep["ekf_raw_z"] - ep["dot_z"]) * 1000
        axes[2, 0].plot(t, err_y, color=c, lw=1.2, alpha=alpha)
        axes[2, 1].plot(t, err_z, color=c, lw=1.2, alpha=alpha)

    # Décoration
    for ax in axes[0]:
        ax.set_ylabel("Estimated position (m)", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title("EKF — Estimated Y position (Kalman)", fontsize=11)
    axes[0, 1].set_title("EKF — Estimated Z position (Kalman)", fontsize=11)

    # Lignes de bruit de référence
    for ax in axes[1]:
        ax.axhline(r_fk_std * 1000, color="blue", lw=1.2, ls="--", alpha=0.8,
                   label=f"σ_FK = {r_fk_std*1000:.0f} mm")
        ax.axhline(r_cam_std * 1000, color="red", lw=1.2, ls=":", alpha=0.8,
                   label=f"σ_cam = {r_cam_std*1000:.0f} mm")
        ax.set_ylabel("Estimated sigma (mm)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[1, 0].set_title("EKF variance, Y axis — P[0,0]^(1/2)", fontsize=11)
    axes[1, 1].set_title("EKF variance, Z axis — P[1,1]^(1/2)", fontsize=11)

    for ax in axes[2]:
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axhline(r_fk_std * 1000, color="blue", lw=1.0, ls="--", alpha=0.6,
                   label=f"±σ_FK = {r_fk_std*1000:.0f} mm")
        ax.axhline(-r_fk_std * 1000, color="blue", lw=1.0, ls="--", alpha=0.6)
        ax.set_ylabel("Erreur EKF (mm)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[2, 0].set_title("EKF residual, Y axis (EKF vs true pos.)", fontsize=11)
    axes[2, 1].set_title("EKF residual, Z axis (EKF vs true pos.)", fontsize=11)

    # Annotation paramètres bruit
    param_text = (
        "EKF parameters:\n"
        f"  Q_pos: sigma = {q_pos_std*1000:.0f} mm  (position process noise)\n"
        f"  Q_vel: sigma = {q_vel_std*1000:.0f} mm/s (velocity process noise)\n"
        f"  R_FK:  sigma = {r_fk_std*1000:.0f} mm  (forward-kinematics measurement noise)\n"
        f"  R_cam: sigma = {r_cam_std*1000:.0f} mm  (KLT camera measurement noise)\n"
        "  State: [y, z, vy, vz]  H = I2 x 0"
    )
    fig.text(0.01, 0.01, param_text, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out = outdir / "03_ekf_kalman.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Plot 4 : Monte Carlo — formes de trajectoires ─────────────────────────────
def plot_monte_carlo(episodes: list[dict], outdir: Path):
    from ur7e_line_follower.target_line import WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_WIDE, dpi=DPI)
    fig.suptitle("Monte Carlo — Stochastic trajectory generation (Catmull-Rom)\n"
                 "10 random rollouts with the trained SAC model",
                 fontsize=15, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (ep, ax) in enumerate(zip(episodes, axes.flat)):
        line = ep["line"]
        c = colors[i]

        wall_rect = plt.Rectangle((WALL_Y_MIN, WALL_Z_MIN),
                                   WALL_Y_MAX - WALL_Y_MIN, WALL_Z_MAX - WALL_Z_MIN,
                                   fill=True, facecolor="#F5F5F5", edgecolor="#BDBDBD", lw=1.5)
        ax.add_patch(wall_rect)

        # Ligne cible (Catmull-Rom)
        ax.plot(line[:, 0], line[:, 1], "--", color=c, lw=1.8, alpha=0.6, label="Cible MC")
        # Trajectoire suivie
        ax.plot(ep["dot_y"], ep["dot_z"], "-", color=c, lw=2.5, alpha=0.9, label="Actual laser")
        ax.plot(ep["dot_y"][0], ep["dot_z"][0], "o", color="k", ms=7, zorder=5)
        ax.plot(ep["dot_y"][-1], ep["dot_z"][-1], "s", color=c, ms=7, zorder=5)

        ax.set_xlim(WALL_Y_MIN - 0.02, WALL_Y_MAX + 0.02)
        ax.set_ylim(WALL_Z_MIN - 0.02, WALL_Z_MAX + 0.02)
        ax.set_aspect("equal")

        level_names = {-1: "Straight", 0: "Gentle", 1: "Moderate", 2: "Random"}
        lname = level_names.get(ep.get("line_level", 2), "MC")
        status = "SUCCESS" if ep["success"] else "FAILURE"
        status_color = C["success"] if ep["success"] else C["failure"]
        ax.set_title(
            f"Episode {i+1} — {lname}\n"
            f"{status}  prog={ep['final_progress']:.0%}",
            fontsize=9, color=status_color, fontweight="bold"
        )

        if i % 5 == 0:
            ax.set_ylabel("Z (m)", fontsize=9)
        if i >= 5:
            ax.set_xlabel("Y (m)", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = outdir / "04_monte_carlo_trajectoires.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Plot 5 : LQR — commandes et manipulabilité ────────────────────────────────
def plot_lqr(episodes: list[dict], outdir: Path):
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE_TALL, dpi=DPI)
    fig.suptitle("LQR Control — Commands and Manipulability\n"
                 "Singularity-adaptive filter on the wall Jacobian",
                 fontsize=16, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ep in enumerate(episodes):
        t = ep["t"]
        c = colors[i]
        a = 0.75

        axes[0, 0].plot(t, ep["act_vy"], color=c, lw=1.5, alpha=a)
        axes[0, 1].plot(t, ep["act_vz"], color=c, lw=1.5, alpha=a)
        axes[1, 0].plot(t, ep["wall_vy"], color=c, lw=1.5, alpha=a)
        axes[1, 1].plot(t, ep["wall_vz"], color=c, lw=1.5, alpha=a)
        axes[2, 0].plot(t, ep["manip_w"], color=c, lw=1.5, alpha=a,
                        label=f"Ep{i+1} {'✓' if ep['success'] else '✗'}")
        axes[2, 1].plot(t, np.abs(ep["act_vy"]) + np.abs(ep["act_vz"]),
                        color=c, lw=1.5, alpha=a)

    W_MIN, W_REF = 0.015, 0.115
    axes[2, 0].axhline(W_REF, color="orange", lw=1.2, ls="--", label=f"W_REF = {W_REF}")
    axes[2, 0].axhline(W_MIN, color="red", lw=1.2, ls=":", label=f"W_MIN = {W_MIN}")
    axes[2, 0].fill_between([0, 15], W_MIN, W_REF, alpha=0.08, color="orange",
                             label="Zone null-space actif")

    labels = [
        ("SAC action — normalized wall vel. vy [-1,1]", "Action vy [norm]"),
        ("SAC action — normalized wall vel. vz [-1,1]", "Action vz [norm]"),
        ("Actual wall velocity vy (after EMA + LQR) [m/s]", "vy (m/s)"),
        ("Actual wall velocity vz (after EMA + LQR) [m/s]", "vz (m/s)"),
        ("Yoshikawa manipulability index", "w = √det(J·Jᵀ)"),
        ("Norme L1 des actions (effort total)", "|vy| + |vz|"),
    ]

    for ax, (title, ylabel) in zip(axes.flat, labels):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.7, ls="--")

    for ax in axes.flat:
        ax.set_xlabel("Time (s)", fontsize=9)

    axes[2, 0].legend(fontsize=8, ncol=2)

    # Annotation LQR
    lqr_text = (
        "LQR gains adaptatifs :\n"
        "  K_i = sqrt(1 + Q/R_i) - 1\n"
        "  R_i in [0.05, 1.0] based on SVD(J_wall)\n"
        "  lambda in [1e-4, 0.08] based on Yoshikawa w\n"
        "  Null-space: alpha=0.03, max=0.05 rad/s"
    )
    fig.text(0.01, 0.01, lqr_text, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#F3E5F5", alpha=0.8))

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    out = outdir / "05_lqr_asservissement.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Plot 6 : Tableau de bord global ──────────────────────────────────────────
def plot_dashboard(episodes: list[dict], outdir: Path):
    fig = plt.figure(figsize=(18, 10), dpi=DPI)
    fig.suptitle("UR7e Line Follower — Dashboard\n"
                 "SAC + HER + EKF + LQR + Monte Carlo  —  10 test episodes",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)
    ax_prog = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_rew  = fig.add_subplot(gs[0, 2])
    ax_klt  = fig.add_subplot(gs[1, 0])
    ax_man  = fig.add_subplot(gs[1, 1])
    ax_bar  = fig.add_subplot(gs[1, 2])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ep_ids = [f"Ep{i+1}" for i in range(10)]

    for i, ep in enumerate(episodes):
        t = ep["t"]
        c = colors[i]
        ax_prog.plot(t, ep["progress"] * 100, color=c, lw=2, alpha=0.85,
                     label=f"Ep{i+1} {'✓' if ep['success'] else '✗'}")
        ax_rmse.plot(t, ep["rmse"] * 100, color=c, lw=1.5, alpha=0.85)
        ax_rew.plot(t, np.cumsum(ep["reward"]), color=c, lw=2, alpha=0.85)
        ax_klt.plot(t, ep["klt_conf"], color=c, lw=1.5, alpha=0.75)
        ax_man.plot(t, ep["manip_w"], color=c, lw=1.5, alpha=0.75)

    # Barres de résultats
    final_progs = [ep["final_progress"] * 100 for ep in episodes]
    success_flags = [ep["success"] for ep in episodes]
    bar_colors = [C["success"] if s else C["failure"] for s in success_flags]
    bars = ax_bar.bar(ep_ids, final_progs, color=bar_colors, edgecolor="k", lw=0.8)
    ax_bar.axhline(98.5, color="green", lw=1.5, ls="--", label="Success threshold 98.5%")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Progression finale (%)", fontsize=10)
    ax_bar.set_title("Progress per episode", fontsize=11)
    ax_bar.legend(fontsize=9)
    ax_bar.tick_params(axis="x", rotation=45)
    for bar, prog in zip(bars, final_progs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{prog:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_prog.axhline(98.5, color="green", lw=1, ls="--", label="Success threshold")
    ax_prog.set_title("Progression sur la ligne (%)", fontsize=11)
    ax_prog.set_ylabel("Progression (%)", fontsize=10)
    ax_prog.legend(fontsize=7, ncol=2)
    ax_prog.grid(True, alpha=0.3)

    ax_rmse.axhline(4, color="orange", lw=1, ls="--", label="RMSE cible 4 cm")
    ax_rmse.set_title("RMSE glissant (30 steps)", fontsize=11)
    ax_rmse.set_ylabel("RMSE (cm)", fontsize=10)
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(True, alpha=0.3)

    ax_rew.set_title("Cumulative reward", fontsize=11)
    ax_rew.set_ylabel("Σ Reward", fontsize=10)
    ax_rew.grid(True, alpha=0.3)

    ax_klt.axhline(0.5, color="orange", lw=1, ls="--")
    ax_klt.set_title("Confiance KLT", fontsize=11)
    ax_klt.set_ylabel("Confiance [0,1]", fontsize=10)
    ax_klt.grid(True, alpha=0.3)

    ax_man.axhline(0.115, color="orange", lw=1, ls="--", label="W_REF")
    ax_man.axhline(0.015, color="red", lw=1, ls=":", label="W_MIN")
    ax_man.set_title("Yoshikawa manipulability", fontsize=11)
    ax_man.set_ylabel("w = √det(JJᵀ)", fontsize=10)
    ax_man.legend(fontsize=8)
    ax_man.grid(True, alpha=0.3)

    for ax in [ax_prog, ax_rmse, ax_rew, ax_klt, ax_man]:
        ax.set_xlabel("Time (s)", fontsize=9)

    n_success = sum(success_flags)
    mean_prog = np.mean(final_progs)
    mean_rmse = np.mean([ep["rmse"][-1] for ep in episodes]) * 100
    stats_text = (
        f"Results — 10 episodes\n"
        f"  Success: {n_success}/10 ({n_success*10}%)\n"
        f"  Progression moyenne : {mean_prog:.1f}%\n"
        f"  RMSE final moyen : {mean_rmse:.1f} cm\n"
        f"  Model: basile_level2 (60k steps)"
    )
    fig.text(0.01, 0.01, stats_text, fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

    out = outdir / "00_dashboard.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")
    return n_success, mean_prog, mean_rmse


# ── Plot 7 : EKF variance bruit blanc détaillé ────────────────────────────────
def plot_ekf_noise_detail(episodes: list[dict], outdir: Path):
    """Courbes de variance + innovation (bruit blanc gaussien)."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    fig.suptitle("EKF — Measurement and Innovation Noise Analysis\n"
                 "Bruit blanc gaussien : processus Q et mesure R",
                 fontsize=15, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Q et R théoriques
    q_pos_std = 0.002
    r_fk_std  = 0.006
    r_cam_std = 0.018

    # Innovation simulée = err_y ~ N(0, R_fk)
    all_innov_y, all_innov_z = [], []
    for ep in episodes:
        innov_y = (ep["ekf_raw_y"] - ep["dot_y"]) * 1000
        innov_z = (ep["ekf_raw_z"] - ep["dot_z"]) * 1000
        all_innov_y.extend(innov_y.tolist())
        all_innov_z.extend(innov_z.tolist())

    # Histogramme des innovations
    bins = 40
    n_y, b_y, _ = axes[0, 0].hist(all_innov_y, bins=bins, density=True,
                                    color="#1565C0", alpha=0.7, label="EKF residuals Y")
    n_z, b_z, _ = axes[0, 1].hist(all_innov_z, bins=bins, density=True,
                                    color="#AD1457", alpha=0.7, label="EKF residuals Z")

    # Courbe gaussienne théorique
    from scipy.stats import norm as sp_norm
    for ax, data, color in [(axes[0, 0], all_innov_y, "#1565C0"),
                             (axes[0, 1], all_innov_z, "#AD1457")]:
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(min(data), max(data), 200)
        ax.plot(x, sp_norm.pdf(x, mu, std), color="k", lw=2.5,
                label=f"N(μ={mu:.1f}, σ={std:.1f}) mm")
        ax.axvline(r_fk_std * 1000, color="blue", ls="--", lw=1.5,
                   label=f"theoretical sigma_FK = {r_fk_std*1000:.0f} mm")
        ax.axvline(-r_fk_std * 1000, color="blue", ls="--", lw=1.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Innovation (mm)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    axes[0, 0].set_title("EKF innovation distribution — Y axis\n(white-noise test)", fontsize=11)
    axes[0, 1].set_title("EKF innovation distribution — Z axis\n(white-noise test)", fontsize=11)

    # Évolution de la covariance P dans le temps
    for i, ep in enumerate(episodes):
        t = ep["t"]
        sig_y_mm = np.sqrt(np.clip(ep["ekf_cov_yy"], 1e-12, None)) * 1000
        sig_z_mm = np.sqrt(np.clip(ep["ekf_cov_zz"], 1e-12, None)) * 1000
        axes[1, 0].plot(t, sig_y_mm, color=colors[i], lw=1.5, alpha=0.8)
        axes[1, 1].plot(t, sig_z_mm, color=colors[i], lw=1.5, alpha=0.8)

    for ax, ax_label in [(axes[1, 0], "Y"), (axes[1, 1], "Z")]:
        ax.axhline(r_fk_std * 1000, color="blue", ls="--", lw=1.5,
                   label=f"R_FK sigma = {r_fk_std*1000:.0f} mm")
        ax.axhline(q_pos_std * 1000, color="green", ls=":", lw=1.5,
                   label=f"Q_pos sigma = {q_pos_std*1000:.0f} mm")
        ax.set_title(f"Kalman std-dev convergence — {ax_label} axis", fontsize=11)
        ax.set_ylabel(f"sigma_{ax_label} (mm)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = outdir / "03b_ekf_bruit_blanc.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        default=str(Path.home() / ".ros/ur7e_line_follower/offline_runs/basile_level2/offline_sac_final.zip"))
    parser.add_argument("--outdir",
        default=str(Path.home() / "presentation_plots"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nUR7e — Presentation plot generation")
    print(f"  Model: {args.model}")
    print(f"  Output: {outdir}")

    print("\n[1/2] Creating environment and loading model...")
    env = OfflineUR7eEnv(reward_profile="normalized_huber", line_level=2, seed=args.seed)
    model = SAC.load(args.model, env=env, device=args.device)
    print("  Model loaded OK")

    # 10 épisodes : mix de niveaux pour variété visuelle
    episode_configs = [
        (-1, args.seed + 0),   # droite fixe
        (0,  args.seed + 1),   # douce
        (0,  args.seed + 2),   # douce
        (1,  args.seed + 3),   # modérée
        (1,  args.seed + 4),   # modérée
        (2,  args.seed + 5),   # aléatoire
        (2,  args.seed + 6),   # aléatoire
        (2,  args.seed + 7),   # aléatoire
        (2,  args.seed + 8),   # aléatoire
        (2,  args.seed + 9),   # aléatoire
    ]

    print("\n[2/2] Running 10 test episodes...")
    episodes = []
    for i, (level, seed) in enumerate(episode_configs):
        ep = run_episode(model, env, seed=seed, line_level=level)
        lnames = {-1: "Straight", 0: "Gentle", 1: "Moderate", 2: "Random"}
        print(f"  Ep{i+1:2d} [{lnames[level]:9s}] : "
              f"{'SUCCESS' if ep['success'] else 'FAILURE':7s}  "
              f"prog={ep['final_progress']:.0%}  "
              f"steps={len(ep['t'])}")
        episodes.append(ep)

    print("\n[3/3] Generating plots...")
    plot_dashboard(episodes, outdir)
    plot_klt(episodes, outdir)
    plot_kinematics(episodes, outdir)
    plot_ekf(episodes, outdir)
    try:
        plot_ekf_noise_detail(episodes, outdir)
    except ImportError:
        print("  [SKIP] scipy not available — EKF white-noise plot skipped")
    plot_monte_carlo(episodes, outdir)
    plot_lqr(episodes, outdir)

    print(f"\nAll PNGs saved to: {outdir}")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
