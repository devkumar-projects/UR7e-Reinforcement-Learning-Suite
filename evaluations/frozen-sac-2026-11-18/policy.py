"""Deterministic SAC actor exported to NumPy; no torch or pickle required.
Inference matches tanh(mu(ReLU(W2 ReLU(W1 observation + b1) + b2))).
Input: 29 float32 values in the original environment units/order. Output: 6
bounded actions in [-1, 1], mapped by the simulator to joint deltas of 0.05 rad.
"""
from pathlib import Path
import numpy as np
class Policy:
    def __init__(self,path=None):
        with np.load(path or Path(__file__).with_name('actor.npz'),allow_pickle=False) as data:
            self.weights={key:data[key].copy() for key in ['w1','b1','w2','b2','wmu','bmu']}
        expected={'w1':(512,29),'b1':(512,),'w2':(512,512),'b2':(512,),'wmu':(6,512),'bmu':(6,)}
        for key,shape in expected.items():
            if self.weights[key].shape!=shape or not np.isfinite(self.weights[key]).all():raise ValueError('Invalid actor array: '+key)
    def predict(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.ndim not in (1,2) or x.shape[-1]!=29 or not np.isfinite(x).all():raise ValueError('Expected finite observation(s) of shape (29,) or (N,29)')
        p=self.weights;x=np.maximum(x@p['w1'].T+p['b1'],0);x=np.maximum(x@p['w2'].T+p['b2'],0)
        return np.tanh(x@p['wmu'].T+p['bmu']).astype(np.float32)
