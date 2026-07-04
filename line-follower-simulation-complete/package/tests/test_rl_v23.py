"""Regression tests for corrected V5 RL mechanics."""
from collections import deque
import numpy as np

from ur7e_line_follower.env import (
    UR7eLineFollowerEnv, PHYSICS_STEPS, SETTLE_STEPS,
    CURRICULUM_MIN_EPISODES,
)
from ur7e_line_follower.target_line import (
    DEFAULT_HOME_DOT, WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX,
    arc_length, curriculum_line_from_start,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _PulseNode:
    def __init__(self):
        self.published = []
        self.waits = []
        self.stop_calls = 0
        self.last_cmd_raw = np.zeros(6)
        self.last_cmd_lqr = np.zeros(6)
        self.last_cmd_null = np.zeros(6)
        self.last_cmd_out = np.zeros(6)
        self._detection_frame_count = 0

    def publish_velocity(self, value):
        value = np.asarray(value, dtype=float).copy()
        self.published.append(value)
        self.last_cmd_raw = value.copy()
        self.last_cmd_lqr = value.copy()
        self.last_cmd_out = value.copy()

    def wait_for_n_steps(self, n_steps, timeout):
        self.waits.append((n_steps, timeout))
        return True

    def wait_for_detection_after(self, frame_count, timeout):
        return True

    def stop(self):
        self.stop_calls += 1

    def get_logger(self):
        return _Logger()


def test_pulse_stops_before_sac_update_and_keeps_diagnostics():
    env = UR7eLineFollowerEnv.__new__(UR7eLineFollowerEnv)
    env.node = _PulseNode()
    env._deterministic_pulse = True
    env._observation_mode = 'privileged_debug'
    cmd = np.arange(6, dtype=float) * 0.01
    assert env._pulse(cmd)
    np.testing.assert_allclose(env.node.published[0], cmd)
    assert env.node.waits == [(PHYSICS_STEPS, 1.0), (SETTLE_STEPS, 0.5)]
    assert env.node.stop_calls == 1
    np.testing.assert_allclose(env._last_cmd_out, cmd)


def test_curriculum_trajectories_are_anchored_and_valid():
    rng = np.random.default_rng(123)
    mean_lateral = []
    for level in (0, 1, 2):
        excursions = []
        for _ in range(40):
            line = curriculum_line_from_start(rng, DEFAULT_HOME_DOT, level=level)
            assert line.shape == (50, 2)
            np.testing.assert_allclose(line[0], DEFAULT_HOME_DOT, atol=1e-5)
            assert arc_length(line) >= 0.50
            assert line[:, 0].min() >= WALL_Y_MIN
            assert line[:, 0].max() <= WALL_Y_MAX
            assert line[:, 1].min() >= WALL_Z_MIN
            assert line[:, 1].max() <= WALL_Z_MAX
            excursions.append(float(np.ptp(line[:, 0])))
        mean_lateral.append(float(np.mean(excursions)))
    assert mean_lateral[0] < mean_lateral[2], mean_lateral


def test_curriculum_advances_by_success_not_steps():
    env = UR7eLineFollowerEnv.__new__(UR7eLineFollowerEnv)
    env._curriculum_enabled = True
    env._curriculum_level_value = 0
    env._recent_successes = deque(maxlen=50)
    env._episode_outcome_recorded = False
    env.node = _PulseNode()
    env._total_steps = 1_000_000
    assert env._curriculum_level() == 0
    for _ in range(CURRICULUM_MIN_EPISODES):
        env._episode_outcome_recorded = False
        env._record_episode_outcome(True)
    assert env._curriculum_level() == 1
    env._curriculum_enabled = False
    assert env._curriculum_level() == 2
