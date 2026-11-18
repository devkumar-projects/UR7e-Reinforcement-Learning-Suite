from pathlib import Path
import json,numpy as np
from ur7e_env_v4 import get_chain,HOME_CONFIG,JOINT_LIMITS
root=Path(__file__).resolve().parent;chain=get_chain();rows=[];joint=0
for i,link in enumerate(chain.links):
 active=bool(chain.active_links_mask[i]);row={'name':link.name,'base_matrix':np.asarray(link.get_link_frame_matrix(0),dtype=float).tolist(),'axis':list(map(float,link.rotation)) if active else None,'joint':joint if active else None}
 if active:joint+=1
 rows.append(row)
assert joint==6
config={'links':rows,'home':HOME_CONFIG.tolist(),'limits':JOINT_LIMITS,'max_delta':0.05,'observation_size':29,'actions':6,'source':'ur7e_generated.urdf in pinned source commit; default kinematics, not robot-specific calibration.'}
(root/'kinematics.json').write_text(json.dumps(config,indent=2)+'\n');print([(r['name'],r['joint']) for r in rows])
