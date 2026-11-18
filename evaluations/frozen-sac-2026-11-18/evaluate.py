"""Frozen SAC vs fixed damped-least-squares position controller. CPU only.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate.py
No training, collision simulation, dynamics, orientation or physical robot access.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,time,hashlib,platform,csv,argparse
from pathlib import Path
import numpy as np
import torch
from checkpoint import load_model, checkpoint_path
from ur7e_traj_env import UR7eTrajEnv, ERREUR_MAX, SEUIL_POINT
from ur7e_env_v4 import fk,JOINT_LIMITS
from trajectory_generator import generate_trajectory,SHAPE_GENERATORS
ROOT=Path(__file__).resolve().parent

def baseline(obs):
    q=obs[:6]*np.array([hi for lo,hi in JOINT_LIMITS]);position=obs[6:9];target=obs[11:14]
    jac=np.empty((3,6));h=1e-4
    for j in range(6):
        dq=np.zeros(6);dq[j]=h;jac[:,j]=(fk(q+dq).astype(float)-fk(q-dq).astype(float))/(2*h)
    # One fixed controller across all tests: DLS damping=0.02, gain=0.8.
    dq=0.8*jac.T@np.linalg.solve(jac@jac.T+0.02**2*np.eye(3),target-position)
    return np.clip(dq/0.05,-1,1).astype(np.float32)

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]

def run(model,controller,trajectory,seed,noise,record=False):
    env=UR7eTrajEnv(fixed_trajectory=trajectory,max_steps=800);obs,_=env.reset(seed=seed)
    rng=np.random.default_rng(seed+600000);path=[];errors=[];latencies=[];progress=[];phase=[];total_reward=0.;max_lateral=0.;info={}
    for step in range(800):
        measured=obs.copy()
        if noise:
            measured[:6]+=rng.normal(0,noise,6)/np.array([hi for lo,hi in JOINT_LIMITS]);measured[6:9]+=rng.normal(0,noise,3)
        st=time.perf_counter_ns()
        action=model.predict(measured,deterministic=True)[0] if controller=='sac' else baseline(measured)
        latencies.append((time.perf_counter_ns()-st)/1e6)
        was_tracking=env._phase==1
        obs,reward,done,trunc,info=env.step(action);total_reward+=reward
        pos=fk(env._angles);path.append(pos.tolist());progress.append(int(env._cursor));phase.append(int(env._phase))
        if was_tracking:errors.append(float(info['err_laterale_mm']));max_lateral=max(max_lateral,errors[-1])
        if done or trunc:break
    final_distance=float(np.linalg.norm(fk(env._angles)-trajectory[-1]))
    reached_end=env._phase==1 and env._cursor==len(trajectory)-1 and final_distance<SEUIL_POINT
    strict=bool(reached_end and done and max_lateral<=ERREUR_MAX*1000)
    reason='completed' if strict else 'timeout' if trunc else 'stagnation' if env._stagnation>150 else 'off_path' if info.get('err_laterale_mm',0)>ERREUR_MAX*1000 else 'endpoint_tolerance_not_met' if info.get('completed',False) else 'other'
    row={'controller':controller,'seed':seed,'noise_position_m_and_joint_rad':noise,'strict_completed':strict,'original_completed_flag':bool(info.get('completed',False)),'acquired':bool(env._phase==1),'stop_reason':reason,'steps':step+1,'progress_fraction':env._cursor/(len(trajectory)-1),'tracking_rms_mm':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,'tracking_samples':len(errors),'final_target_distance_mm':final_distance*1000,'max_tracking_error_mm':max_lateral if errors else None,'reward':total_reward,'action_latency_median_ms':float(np.median(latencies)),'action_latency_p95_ms':float(np.percentile(latencies,95))}
    replay={'target':trajectory.tolist(),'path':path,'cursor':progress,'phase':phase,'result':row} if record else None
    env.close();return row,replay

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--per-shape',type=int,default=20);args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    model=load_model()
    assert model.observation_space.shape==(29,) and model.action_space.shape==(6,)
    out=ROOT/'evaluation';out.mkdir(exist_ok=True);rows=[];replays=[];started=time.monotonic()
    # Test manifest fixed before evaluation; no model selection/tuning on these trajectories.
    cases=[(shape,2026111800+100*index+i) for index,shape in enumerate(SHAPE_GENERATORS) for i in range(args.per_shape)]
    (out/'test-manifest.json').write_text(json.dumps({'cases':[{'shape':sh,'seed':se} for sh,se in cases],'noises':[0.,.002,.005],'controllers':['sac','dls'],'max_steps':800,'completion_tolerance_m':float(SEUIL_POINT),'tracking_limit_m':float(ERREUR_MAX)},indent=2)+'\n')
    for noise in [0.,.002,.005]:
        for controller in ['sac','dls']:
            for shape,seed in cases:
                traj,_=generate_trajectory(shape=shape,n_points=60,seed=seed)
                row,replay=run(model,controller,traj,seed,noise,record=noise==0 and seed%100 in [0,1,2,3])
                row['shape']=shape;rows.append(row)
                if replay:replay['shape']=shape;replays.append(replay)
            print(json.dumps({'noise':noise,'controller':controller,'episodes':len(cases),'strict_completed':sum(r['strict_completed'] for r in rows[-len(cases):]),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
            (out/'episodes.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for noise in [0.,.002,.005]:
        for controller in ['sac','dls']:
            rs=[r for r in rows if r['controller']==controller and r['noise_position_m_and_joint_rad']==noise];tracked=[r['tracking_rms_mm'] for r in rs if r['tracking_rms_mm'] is not None];completed=[r['tracking_rms_mm'] for r in rs if r['strict_completed'] and r['tracking_rms_mm'] is not None];k=sum(r['strict_completed'] for r in rs)
            summaries.append({'controller':controller,'sensor_noise':noise,'episodes':len(rs),'strict_completed':k,'completion_rate':k/len(rs),'completion_wilson_95':wilson(k,len(rs)),'acquisition_rate':sum(r['acquired'] for r in rs)/len(rs),'mean_tracking_rms_mm_acquired':float(np.mean(tracked)) if tracked else None,'mean_tracking_rms_mm_completed':float(np.mean(completed)) if completed else None,'original_flag_false_positives':sum(r['original_completed_flag'] and not r['strict_completed'] for r in rs),'mean_progress_fraction':float(np.mean([r['progress_fraction'] for r in rs])),'median_episode_action_latency_ms':float(np.median([r['action_latency_median_ms'] for r in rs])),'stops':{reason:sum(r['stop_reason']==reason for r in rs) for reason in ['completed','timeout','stagnation','off_path','endpoint_tolerance_not_met','other']}})
    versions={m:__import__(m).__version__ for m in ['numpy','torch','stable_baselines3','gymnasium','ikpy','matplotlib']}
    summary={'date':'2026-11-18','checkpoint_sha256':hashlib.sha256(checkpoint_path().read_bytes()).hexdigest(),'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'device':'cpu','torch_threads':1,'elapsed_seconds':round(time.monotonic()-started,2),'protocol':'100 prespecified seeded trajectories, balanced over five shape generators; paired controllers and sensor-noise conditions; no retraining. Training seed history unavailable: exact disjointness from original training cannot be certified.','conditions':summaries}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(out/'replays.json').write_text(json.dumps(replays,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
