"""Portable NumPy FK from exported original IKPy link transforms."""
from pathlib import Path
import json,numpy as np
CONFIG=json.loads(Path(__file__).with_name('kinematics.json').read_text())
HOME_CONFIG=np.array(CONFIG['home'],dtype=np.float32)
JOINT_LIMITS=CONFIG['limits']
def chain_points(q):
    t=np.eye(4);points=[]
    for link in CONFIG['links']:
        t=t@np.array(link['base_matrix'])
        if link['joint'] is not None:
            x,y,z=link['axis'];a=float(q[link['joint']]);c=np.cos(a);s=np.sin(a);v=1-c
            r=np.eye(4);r[:3,:3]=[[x*x*v+c,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,y*y*v+c,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,z*z*v+c]];t=t@r
        points.append(t[:3,3].copy())
    return points
def fk(q):return chain_points(q)[-1].astype(np.float32)
