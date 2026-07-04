"""Tests purs de la commande cartésienne 2D V2.2."""
import numpy as np

from ur7e_line_follower.control import (
    ACTION_SPACE_DIM, MAX_WALL_SPEED_M_S, MAX_JOINT_SPEED_RAD_S,
    normalized_action_to_wall_velocity, wall_action_to_joint_velocity,
)
HOME_POSITIONS = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])
from ur7e_line_follower.kinematics import wall_jacobian


def test_action_dimension_is_2():
    assert ACTION_SPACE_DIM == 2


def test_normalized_action_speed_is_bounded():
    v = normalized_action_to_wall_velocity(np.array([1.0, 1.0]), None)
    assert v.shape == (2,)
    assert np.linalg.norm(v) <= MAX_WALL_SPEED_M_S + 1e-12


def test_action_filter_reduces_first_step():
    v = normalized_action_to_wall_velocity(np.array([1.0, 0.0]), np.zeros(2))
    assert 0.0 < v[0] < MAX_WALL_SPEED_M_S
    assert abs(v[1]) < 1e-12


def test_wall_action_maps_to_six_joint_velocities():
    qdot, wall_v = wall_action_to_joint_velocity(HOME_POSITIONS, np.array([0.5, -0.25]), np.zeros(2))
    assert qdot.shape == (6,)
    assert wall_v.shape == (2,)
    assert np.all(np.isfinite(qdot))
    assert np.linalg.norm(qdot) <= MAX_JOINT_SPEED_RAD_S + 1e-9


def test_mgi_output_moves_spot_in_requested_direction():
    qdot, wall_v = wall_action_to_joint_velocity(HOME_POSITIONS, np.array([0.4, 0.2]), np.zeros(2))
    achieved = wall_jacobian(HOME_POSITIONS) @ qdot
    assert np.dot(achieved, wall_v) > 0.0


def test_command_hold_produces_measurable_joint_increment():
    """À HOME, une action franche tenue 0,10 s doit dépasser 1 mrad."""
    qdot, _ = wall_action_to_joint_velocity(
        HOME_POSITIONS, np.array([0.8, 0.0]), np.zeros(2))
    dq = qdot * 0.10
    assert np.linalg.norm(dq) > 1e-3
