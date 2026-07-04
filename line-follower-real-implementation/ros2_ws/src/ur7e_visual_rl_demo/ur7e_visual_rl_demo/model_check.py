from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import rclpy
from rclpy.node import Node
from stable_baselines3 import SAC


class ModelCheck(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_model_check')
        self.declare_parameter('model_path', '')
        self.declare_parameter('expected_sha256', '')

    def run(self) -> bool:
        path = Path(str(self.get_parameter('model_path').value)).expanduser().resolve()
        if not path.is_file():
            print(f'[MODEL FAIL] missing: {path}')
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(self.get_parameter('expected_sha256').value).strip().lower()
        if expected and digest != expected:
            print(f'[MODEL FAIL] sha256 mismatch\nactual={digest}\nexpected={expected}')
            return False
        model = SAC.load(str(path), device='cpu')
        obs_shape = tuple(model.observation_space.shape)
        act_shape = tuple(model.action_space.shape)
        if obs_shape != (33,) or act_shape != (2,):
            print(f'[MODEL FAIL] contract obs={obs_shape}, action={act_shape}')
            return False
        action, _ = model.predict(np.zeros(33, dtype=np.float32), deterministic=True)
        print('[MODEL PASS]')
        print('path:', path)
        print('sha256:', digest)
        print('observation_shape:', obs_shape)
        print('action_shape:', act_shape)
        print('action_on_zero:', np.asarray(action).tolist())
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModelCheck()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        raise SystemExit(2)
