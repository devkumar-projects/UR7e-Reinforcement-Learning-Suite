"""Static and pure-Python contracts for the V3.1 merged package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase1_console_tools_are_registered():
    setup = (ROOT / 'setup.py').read_text()
    assert 'phase1_repeatability' in setup
    assert 'phase1_action_directions' in setup


def test_train_exposes_reward_ablation_entropy_and_tripwire():
    train = (ROOT / 'ur7e_line_follower' / 'train.py').read_text()
    assert "--reward-profile" in train
    assert "--ent-coef" in train
    assert "--tripwire-early-stop" in train
    assert 'class TripwireCallback' in train


def test_demos_accept_only_explicit_success():
    train = (ROOT / 'ur7e_line_follower' / 'train.py').read_text()
    assert "success = bool(info.get('is_success', False))" in train
    assert "or terminated" not in train.split('def generate_demos', 1)[1].split('def inject_demos', 1)[0]


def test_record_bonus_no_longer_duplicates_progress_reward():
    env = (ROOT / 'ur7e_line_follower' / 'env.py').read_text()
    section = env.split('if (dist_ordered <= TRACKING_GATE_M', 1)[1].split(
        'if delta_s > STAGNATION_PROGRESS_EPS_M', 1)[0]
    assert 'reward += record_bonus' not in section
