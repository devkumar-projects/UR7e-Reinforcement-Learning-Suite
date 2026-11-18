"""Export the deterministic actor from the original trusted checkpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import numpy as np
import torch
from checkpoint import load_model
from ur7e_traj_env import UR7eTrajEnv
from trajectory_generator import generate_trajectory
from policy import Policy
root=Path(__file__).resolve().parent;torch.set_num_threads(1);m=load_model();sd=m.actor.state_dict()
keys={'w1':'latent_pi.0.weight','b1':'latent_pi.0.bias','w2':'latent_pi.2.weight','b2':'latent_pi.2.bias','wmu':'mu.weight','bmu':'mu.bias'}
np.savez_compressed(root/'actor.npz',**{k:sd[v].cpu().numpy() for k,v in keys.items()});p=Policy()
errors=[];observations=[]
for seed in range(2026111700,2026111705):
 traj,_=generate_trajectory(seed=seed);e=UR7eTrajEnv(fixed_trajectory=traj);o,_=e.reset()
 for i in range(100):
  a=m.predict(o,deterministic=True)[0];na=p.predict(o);errors.append(float(np.max(np.abs(a-na))));observations.append(o)
  o,r,d,t,info=e.step(a)
  if d or t:break
 e.close()
batch=np.array(observations,dtype=np.float32);expected=np.array([m.predict(o,deterministic=True)[0] for o in batch]);batcherr=float(np.max(np.abs(p.predict(batch)-expected)))
assert max(errors)<1e-4 and batcherr<1e-4,(max(errors),batcherr)
result={'samples':len(errors),'single_max_abs_action_difference':max(errors),'batch_max_abs_action_difference':batcherr,'tolerance':1e-4,'reference':'Original Stable-Baselines3 deterministic SAC.predict on CPU','format':'NumPy NPZ; allow_pickle=False','export':'Deterministic mean actor only; original stochastic policy and optimizers retained in best_model.zip.'}
(root/'evaluation/actor-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
