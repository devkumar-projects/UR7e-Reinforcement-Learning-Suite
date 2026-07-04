#!/usr/bin/env python3
"""Lightweight terminal dashboard for monitored UR7e offline SAC runs.

No ROS dependency. Reads JSON/CSV artifacts produced by
`offline_train_ur7e_curriculum_monitored.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

BASE = Path.home() / ".ros" / "ur7e_line_follower" / "offline_runs"
LEVELS = [
    ("fixed_line", "Fixe"),
    ("curriculum_level_0_gentle", "Doux"),
    ("curriculum_level_1_moderate", "Modéré"),
    ("monte_carlo_random_lines", "Aléatoire"),
]


def latest_run() -> Path | None:
    if not BASE.exists():
        return None
    runs = [p for p in BASE.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def fmt_pct(v):
    try:
        return f"{100.0*float(v):5.1f}%"
    except Exception:
        return "  n/a "


def fmt_cm(v):
    try:
        return f"{100.0*float(v):5.2f} cm"
    except Exception:
        return "   n/a  "


def render(run: Path):
    summary = load_json(run / "training_summary.json") or {}
    evaluation = load_json(run / "evaluation_latest.json") or {}
    results = evaluation.get("results", {})
    checkpoints = sorted((run / "checkpoints").glob("*.zip")) if (run / "checkpoints").exists() else []

    os.system("clear")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          UR7e OFFLINE SAC — SUIVI D'ENTRAÎNEMENT               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"Run       : {run}")
    print(f"Timesteps : {summary.get('timestep', 0):,}")
    print(f"Épisodes  : {summary.get('episodes', 0):,}")
    print(f"Niveau    : {summary.get('curriculum_level', 'fixe')}")
    print(f"Device    : {summary.get('device', 'n/a')}")
    print(f"Checkpoints disponibles : {len(checkpoints)}")
    if checkpoints:
        print(f"Dernier checkpoint      : {checkpoints[-1].name}")

    rolling = summary.get("rolling_50", {})
    print("\nMÉTRIQUES D'ENTRAÎNEMENT — FENÊTRE 50 ÉPISODES")
    success = float(rolling.get("success_rate", 0.0) or 0.0)
    progress = float(rolling.get("mean_progress", 0.0) or 0.0)
    rmse = rolling.get("mean_recent_rmse_m", float("nan"))
    ret = rolling.get("mean_return", float("nan"))
    ep_len = rolling.get("mean_episode_length", float("nan"))
    print(f"Succès      : [{bar(success)}] {fmt_pct(success)}")
    print(f"Progression : [{bar(progress)}] {fmt_pct(progress)}")
    print(f"RMSE        : {fmt_cm(rmse)}")
    try:
        print(f"Return      : {float(ret):+.3f}")
        print(f"Durée ep.   : {float(ep_len):.1f} steps")
    except Exception:
        pass

    print("\nDERNIÈRE ÉVALUATION GLOBALE")
    if not results:
        print("Aucune évaluation périodique disponible pour l'instant.")
    else:
        print(f"Évaluation au timestep : {evaluation.get('timestep', 'n/a')}")
        print("Niveau       Succès   Progression   RMSE      Return")
        print("──────────  ───────  ───────────  ────────  ─────────")
        for key, label in LEVELS:
            m = results.get(key, {})
            try:
                print(
                    f"{label:<10}  {fmt_pct(m.get('success_rate')):>7}  "
                    f"{fmt_pct(m.get('mean_progress')):>11}  "
                    f"{fmt_cm(m.get('mean_recent_rmse_m')):>8}  "
                    f"{float(m.get('mean_return', float('nan'))):+9.3f}"
                )
            except Exception:
                print(f"{label:<10}  n/a")

    plot_dir = run / "plots"
    print("\nFICHIERS DE SUIVI")
    print(f"TensorBoard : {run / 'tb'}")
    print(f"Courbes PNG : {plot_dir}")
    print(f"Snapshots   : {run / 'evaluation_snapshots'}")
    print(f"CSV épisodes: {run / 'episode_metrics.csv'}")
    print(f"CSV global  : {run / 'evaluation_history.csv'}")
    print("\nRafraîchissement automatique — Ctrl+C pour quitter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run directory; latest run by default")
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        run = Path(args.run).expanduser() if args.run else latest_run()
        if run is None or not run.exists():
            os.system("clear")
            print(f"Aucun run trouvé dans {BASE}")
        else:
            render(run)
        if args.once:
            break
        time.sleep(max(1.0, args.refresh))


if __name__ == "__main__":
    main()
