"""Static contracts preventing accidental long default training and per-step gz calls."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_requires_explicit_cli_arguments():
    text = (ROOT / 'ur7e_line_follower' / 'train.py').read_text()
    assert 'if len(sys.argv) == 1' in text
    assert 'aucun argument reçu' in text


def test_environment_has_no_per_step_laser_gz_update():
    text = (ROOT / 'ur7e_line_follower' / 'env.py').read_text()
    step = text.split('    def step(self, action: np.ndarray):', 1)[1]
    step = step.split('    def close(', 1)[0] if '    def close(' in step else step
    assert 'update_laser_dot_visual()' not in step


def test_training_uses_image_overlay_not_gz_pose():
    text = (ROOT / 'ur7e_line_follower' / 'train.py').read_text()
    assert 'update_dot_visual=False' in text
    launch = (ROOT / 'launch' / 'simulation.launch.py').read_text()
    assert "'use_sim_laser_overlay': True" in launch
