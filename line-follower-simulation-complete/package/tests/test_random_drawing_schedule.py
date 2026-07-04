from pathlib import Path
import numpy as np

from ur7e_line_follower.target_line import (
    random_line_from_start, DEFAULT_HOME_DOT, arc_length,
    WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX,
)


def test_control_smoke_displays_atomic_scene():
    src = (Path(__file__).parents[1] / 'ur7e_line_follower' / 'control_smoke.py').read_text()
    assert 'load_current_trajectory' in src
    assert 'dessin runtime déjà présent' in src
    assert 'show_trajectory_with_retry' not in src


def test_trials_per_drawing_is_configurable():
    env_src = (Path(__file__).parents[1] / 'ur7e_line_follower' / 'env.py').read_text()
    train_src = (Path(__file__).parents[1] / 'ur7e_line_follower' / 'train.py').read_text()
    assert 'trials_per_trajectory' in env_src
    assert '--trials-per-drawing' in train_src


def test_random_drawing_starts_at_home_dot():
    line = random_line_from_start(np.random.default_rng(4), DEFAULT_HOME_DOT)
    assert np.linalg.norm(line[0] - DEFAULT_HOME_DOT) < 1e-6
    assert arc_length(line) >= 0.50


def test_random_drawing_inside_wall():
    rng = np.random.default_rng(12)
    for _ in range(30):
        line = random_line_from_start(rng, DEFAULT_HOME_DOT)
        assert line[:, 0].min() >= WALL_Y_MIN
        assert line[:, 0].max() <= WALL_Y_MAX
        assert line[:, 1].min() >= WALL_Z_MIN
        assert line[:, 1].max() <= WALL_Z_MAX


def test_launch_generates_runtime_world_before_gazebo():
    src = (Path(__file__).parents[1] / 'launch' / 'simulation.launch.py').read_text()
    assert '_write_runtime_world' in src
    assert 'inject_trajectory_into_world' in src
    assert 'RUNTIME_WORLD' in src
