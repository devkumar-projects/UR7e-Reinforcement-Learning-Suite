"""
Tests de l'invariant obs[12:15] — appelle le code de production de env.py.

Stratégie :
  - Instancier UR7eLineFollowerEnv.__new__() pour éviter __init__ (ROS/Gazebo).
  - Initialiser manuellement les attributs requis par _get_obs().
  - Monkeypatcher fk_ur et manipulability_obs (pas de URDF nécessaire).
  - Appeler env._get_obs() et inspecter obs[12:15].

Cela teste le vrai chemin de code (_get_obs), pas une copie dans le test.
"""
import sys
import pathlib

import numpy as np
import pytest
from gymnasium import spaces

_pkg_root = pathlib.Path(__file__).resolve().parents[1]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import ur7e_line_follower.env as env_mod
from ur7e_line_follower.env import (
    UR7eLineFollowerEnv, OBSERVATION_SCHEMA_VERSION, OBSERVATION_SPACE_DIM,
)

print(f"[obs_invariant tests] env module: {env_mod.__file__}")


# ── Stubs minimaux ────────────────────────────────────────────────────────────

class _FakeEKF:
    position    = np.zeros(2, dtype=np.float32)
    uncertainty = np.ones(2, dtype=np.float32) * 0.01
    nis         = 0.0


class _FakeNode:
    joint_pos     = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])
    joint_vel     = np.zeros(6)
    cam_detection = np.zeros(7, dtype=np.float32)
    cam_guidance = np.zeros(3, dtype=np.float32)
    ekf           = _FakeEKF()

    def get_laser_dot(self):
        return np.array([0.05, 0.60])


def _make_env(observation_mode: str) -> UR7eLineFollowerEnv:
    env = UR7eLineFollowerEnv.__new__(UR7eLineFollowerEnv)
    env._observation_mode      = observation_mode
    env._sensor_noise          = False
    env._joint_noise_sigma_rad = 0.0
    env._fk_noise_sigma_m      = 0.0
    env._cam_noise_sigma_m     = 0.0
    env.waypoints  = np.array([[0.1, 0.3], [0.2, 0.5], [0.0, 0.7]], dtype=np.float64)
    env._n_wp      = 3
    env._wp_idx    = 1
    env._last_action = np.array([0.2, -0.3], dtype=np.float64)
    env._last_wall_velocity = np.array([0.012, -0.024], dtype=np.float64)
    env.observation_space = spaces.Box(low=-2.0, high=2.0,
                                       shape=(OBSERVATION_SPACE_DIM,), dtype=np.float32)
    env.node       = _FakeNode()
    env.np_random  = np.random.default_rng(0)
    return env


@pytest.fixture(autouse=True)
def patch_kinematics(monkeypatch):
    """Remplace fk_ur et manipulability_obs par des stubs sans URDF."""
    monkeypatch.setattr(env_mod, 'fk_ur', lambda q: np.array([0.9, 0.05, 0.6]))
    monkeypatch.setattr(env_mod, 'manipulability_obs', lambda q: np.zeros(3, dtype=np.float32))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_obs_schema_version():
    assert OBSERVATION_SCHEMA_VERSION == 5
    assert OBSERVATION_SPACE_DIM == 33


def test_real_mode_obs_shape():
    env = _make_env('real')
    obs = env._get_obs()
    assert obs.shape == (33,), f"shape={obs.shape}"


def test_real_mode_uses_visual_guidance():
    """Mode real → obs[12:15] reprend uniquement /line_guidance."""
    env = _make_env('real')
    env.node.cam_guidance = np.array([0.25, -0.40, 0.60], dtype=np.float32)
    obs = env._get_obs()
    np.testing.assert_allclose(obs[12:15], env.node.cam_guidance, atol=1e-6)



def test_zero_mode_slots_zero():
    """Mode 'zero' → obs[12:15] = [0, 0, 0]."""
    env = _make_env('zero')
    obs = env._get_obs()
    assert np.allclose(obs[12:15], 0.0), \
        f"Mode zero: obs[12:15]={obs[12:15]}"


def test_privileged_debug_nonzero():
    """Mode 'privileged_debug' → obs[12:15] non-nul avec wp_idx=1."""
    env = _make_env('privileged_debug')
    env.node.cam_detection = np.zeros(7, dtype=np.float32)  # force analytical path
    obs = env._get_obs()
    assert obs.shape == (33,), f"shape={obs.shape}"
    # wp_idx=1, dot=[0.05,0.60], wp[1]=[0.2,0.5] → priv_goal != 0
    assert not np.allclose(obs[12:15], 0.0), \
        f"privileged_debug: obs[12:15]={obs[12:15]} devrait être non-nul"


def test_real_no_detection_cam_zeros():
    """Mode 'real' sans détection → obs[15:22] = 0."""
    env = _make_env('real')
    env.node.cam_detection = np.zeros(7, dtype=np.float32)
    obs = env._get_obs()
    assert np.allclose(obs[15:22], 0.0), \
        f"Mode real sans détection: obs[15:22]={obs[15:22]}"


def test_real_with_detection_cam_filled():
    """Mode 'real' avec ligne+laser visibles → obs[15:22] reprend cam_detection."""
    env = _make_env('real')
    cam = np.array([1.0, 0.12, 0.9, 0.5, 0.3, 0.7, 1.0], dtype=np.float32)
    env.node.cam_detection = cam.copy()
    obs = env._get_obs()
    np.testing.assert_allclose(obs[15:22], cam, atol=1e-5)


def test_zero_mode_ignores_live_camera():
    """Le mode zero reste nul même si /line_detection contient une détection valide."""
    env = _make_env('zero')
    env.node.cam_detection = np.array([1.0, 0.4, 0.9, 0.2, 0.8, 0.7, 1.0], dtype=np.float32)
    obs = env._get_obs()
    assert np.allclose(obs[15:22], 0.0), obs[15:22]


def test_privileged_debug_ignores_live_camera():
    """Le mode privilégié utilise toujours la caméra analytique, jamais le topic ROS."""
    env = _make_env('privileged_debug')
    live = np.array([1.0, -0.99, 0.01, -0.5, 0.5, 0.02, 1.0], dtype=np.float32)
    env.node.cam_detection = live
    obs = env._get_obs()
    assert not np.allclose(obs[15:22], live), "fuite du flux réel en privileged_debug"
    assert obs[15] == pytest.approx(1.0)


def test_real_preserves_partial_detection():
    """Une ligne visible sans laser reste informative et ne doit pas être remplacée par 7 zéros."""
    env = _make_env('real')
    cam = np.array([1.0, 0.0, 0.75, 1.0, 0.0, 0.6, 0.0], dtype=np.float32)
    env.node.cam_detection = cam
    obs = env._get_obs()
    np.testing.assert_allclose(obs[15:22], cam, atol=1e-6)


def test_obs_clipped_in_bounds():
    """obs est clampé dans [-2, 2]."""
    env = _make_env('real')
    obs = env._get_obs()
    assert np.all(obs >= -2.0) and np.all(obs <= 2.0), \
        f"obs hors [-2,2]: min={obs.min():.3f} max={obs.max():.3f}"


def test_assert_fires_on_slot_leak(monkeypatch):
    """L'assert dans _get_obs() lève AssertionError si obs[12:15] fuite en mode zero."""
    env = _make_env('zero')

    # Patch np.concatenate pour injecter une valeur parasite dans le slot 13
    orig_concat = np.concatenate

    def _patched(arrays, **kw):
        result = orig_concat(arrays, **kw).copy()
        result[13] = 0.5   # fuite artificielle
        return result

    monkeypatch.setattr(env_mod.np, 'concatenate', _patched)
    with pytest.raises(AssertionError):
        env._get_obs()


def test_action_schema_is_cartesian_2d():
    from ur7e_line_follower.control import ACTION_SPACE_DIM, CONTROL_SCHEMA_VERSION
    assert ACTION_SPACE_DIM == 2
    assert CONTROL_SCHEMA_VERSION == 2


def test_markov_control_state_is_observed():
    env = _make_env('real')
    obs = env._get_obs()
    np.testing.assert_allclose(obs[29:31], env._last_action, atol=1e-6)
    from ur7e_line_follower.control import MAX_WALL_SPEED_M_S
    np.testing.assert_allclose(
        obs[31:33], env._last_wall_velocity / MAX_WALL_SPEED_M_S, atol=1e-6)
