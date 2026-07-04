"""Shared constants and helpers for the guarded real-robot path."""
from __future__ import annotations

import csv
import hashlib
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Training contract from V3.3.
TRAINING_HOME = np.array(
    [-0.133, -1.5707963267948966, 1.5707963267948966,
     0.0, 1.5707963267948966, 0.0],
    dtype=np.float64,
)
TRAINING_Q_LOW = np.array(
    [-2.0 * np.pi, -np.pi, 0.0, -2.0 * np.pi, -2.0 * np.pi, -2.0 * np.pi],
    dtype=np.float64,
)
TRAINING_Q_HIGH = np.array(
    [2.0 * np.pi, 0.0, np.pi, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
    dtype=np.float64,
)
PHYSICAL_Q_LOW = np.full(6, -2.0 * np.pi, dtype=np.float64)
PHYSICAL_Q_HIGH = np.full(6, 2.0 * np.pi, dtype=np.float64)

TRAINING_HOME_TCP = np.array([0.50513336, 0.06690603, 0.4878], dtype=np.float64)
TRAINING_HOME_DOT = np.array([0.00069792, 0.4878], dtype=np.float64)

OBS_DIM = 33
ACTION_DIM = 2
TRAINING_MAX_WALL_SPEED_M_S = 0.12
MAX_DOT_DIST_M = 0.50

WALL_Y_MIN = -0.65
WALL_Y_MAX = 0.65
WALL_Z_MIN = 0.20
WALL_Z_MAX = 1.30


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q_normalized(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    out = 2.0 * (q - TRAINING_Q_LOW) / (TRAINING_Q_HIGH - TRAINING_Q_LOW) - 1.0
    return np.clip(out, -1.0, 1.0)


def dot_is_on_wall(dot: np.ndarray | None, margin: float = 0.02) -> bool:
    if dot is None:
        return False
    y, z = map(float, np.asarray(dot).reshape(2))
    return (
        WALL_Y_MIN + margin <= y <= WALL_Y_MAX - margin
        and WALL_Z_MIN + margin <= z <= WALL_Z_MAX - margin
    )


def min_joint_margin(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    return float(np.min(np.minimum(q - PHYSICAL_Q_LOW, PHYSICAL_Q_HIGH - q)))


def vector_age(last_mono: float) -> float:
    return 999.0 if last_mono <= 0.0 else max(0.0, time.monotonic() - last_mono)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def finite(values: Iterable[float]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(list(values), dtype=np.float64))))


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    n = float(np.linalg.norm(v))
    if n > max_norm > 0.0:
        v *= max_norm / n
    return v


def wrapped_joint_delta(q: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
    """Shortest signed angular displacement from q_reference to q."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    ref = np.asarray(q_reference, dtype=np.float64).reshape(6)
    return (q - ref + np.pi) % (2.0 * np.pi) - np.pi
