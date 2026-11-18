"""Load the upstream SAC checkpoint on CPU, discarding saved training state.
The original archive includes NumPy-2 pickled observations. Override these
unused training buffers and reconstruct the documented Gymnasium spaces to
support NumPy 1.26 as well. Learned tensors and architecture are unchanged.
Only load trusted original checkpoints. For portable inference use policy.py.
"""
from pathlib import Path
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

def checkpoint_path(path=None):
    if path is not None:
        return Path(path)
    default = Path(__file__).with_name('best_model.zip')
    if not default.exists():
        default = Path(__file__).resolve().parents[2]/'line-follower-simulation-simple/models_traj/best_model.zip'
    return default

def load_model(path=None):
    return SAC.load(checkpoint_path(path),device='cpu',custom_objects={
        'observation_space':spaces.Box(low=-np.inf,high=np.inf,shape=(29,),dtype=np.float32),
        'action_space':spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32),
        '_last_obs':None,'_last_episode_starts':None,'_last_original_obs':None,
        'ep_info_buffer':None,'ep_success_buffer':None,
    })
