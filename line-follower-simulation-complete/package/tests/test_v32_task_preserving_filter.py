import numpy as np
from ur7e_line_follower.control import wall_action_to_joint_velocity
from ur7e_line_follower.kinematics import wall_jacobian
from ur7e_line_follower.singularity import command_filter_diagnostics, null_space_manip_correction

HOME = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])


def test_null_term_disabled_when_wall_task_is_well_conditioned():
    np.testing.assert_allclose(null_space_manip_correction(HOME), 0.0, atol=1e-12)


def test_filtered_commands_preserve_four_wall_directions():
    J = wall_jacobian(HOME)
    actions = {
        '+y': (np.array([1.0, 0.0]), 0, +1),
        '-y': (np.array([-1.0, 0.0]), 0, -1),
        '+z': (np.array([0.0, 1.0]), 1, +1),
        '-z': (np.array([0.0, -1.0]), 1, -1),
    }
    for _, (action, axis, sign) in actions.items():
        qdot, _ = wall_action_to_joint_velocity(HOME, action, np.zeros(2), max_speed=0.054)
        diag = command_filter_diagnostics(HOME, qdot, q_dot_max=0.35)
        achieved = J @ np.asarray(diag['out_cmd'])
        assert achieved[axis] * sign > 0.0
        assert abs(achieved[axis]) >= 2.0 * abs(achieved[1-axis])
