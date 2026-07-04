"""Persist and share the trajectory currently displayed in Gazebo."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

STORE_DIR  = Path.home() / '.ros' / 'ur7e_line_follower'
STORE_PATH = STORE_DIR / 'current_trajectory.npy'
MODEL_NAME_PATH = STORE_DIR / 'current_trajectory_model.json'
LAUNCH_MODEL_NAME = 'trajectory_visual'


def save_current_trajectory(waypoints: np.ndarray) -> Path:
    wp = np.asarray(waypoints, dtype=np.float32)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        raise ValueError('trajectory must have shape (N,2)')
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix('.tmp.npy')
    np.save(tmp, wp)
    tmp.replace(STORE_PATH)
    return STORE_PATH


def load_current_trajectory() -> np.ndarray | None:
    try:
        wp = np.load(STORE_PATH)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2 or not np.all(np.isfinite(wp)):
        return None
    return wp.astype(np.float32)


def save_current_model_name(model_name: str) -> None:
    """Persiste le nom du modèle Gazebo actuellement affiché."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_NAME_PATH.with_suffix('.tmp.json')
    tmp.write_text(json.dumps({'model_name': str(model_name)}), encoding='utf-8')
    tmp.replace(MODEL_NAME_PATH)


def load_current_model_name() -> str:
    """Retourne le nom persisté, ou le nom du modèle injecté par le launch."""
    try:
        data = json.loads(MODEL_NAME_PATH.read_text(encoding='utf-8'))
        name = str(data.get('model_name', LAUNCH_MODEL_NAME)).strip()
        return name if name else LAUNCH_MODEL_NAME
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
        return LAUNCH_MODEL_NAME
