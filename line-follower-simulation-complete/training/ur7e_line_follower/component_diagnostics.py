"""Diagnostics offline : MGD/MGI, Kalman, LQR et Monte-Carlo.

La mesure caméra absolue utilisée ici est une étude de calibration potentielle ;
le KLT relatif de la V2.2 n’est pas injecté directement dans le Kalman.
"""
from __future__ import annotations

import csv
import argparse
import time
from pathlib import Path
import numpy as np

from .kinematics import (fk_ur, laser_wall_dot, jacobian, wall_jacobian,
                         cartesian_to_joint_vel, wall_velocity_to_joint_vel,
                         jacobian_condition)
from .ekf import LaserDotEKF
from .singularity import yoshikawa, lqr_gains, lqr_velocity_correction, null_space_manip_correction
from .target_line import random_line, arc_length, WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX

HOME_POSITIONS = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0], dtype=np.float64)

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'
TEST_DIR = DATA_DIR / 'tests'
TEST_DIR.mkdir(parents=True, exist_ok=True)


def _writer(path, fields):
    f = open(path, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def run_kinematics(rng, tag, n=400):
    path = TEST_DIR / f'kinematics_{tag}.csv'
    f, w = _writer(path, ['sample','q0','q1','q2','q3','q4','q5','tcp_x','tcp_y','tcp_z','dot_y','dot_z','on_wall','cond_tcp','cond_wall','cart_vel_err','wall_vel_err'])
    for i in range(n):
        q = HOME_POSITIONS + rng.normal(0.0, [0.3,0.25,0.25,0.4,0.35,0.5])
        q[2] = np.clip(q[2], 0.05, np.deg2rad(155))
        tcp = fk_ur(q); dot = laser_wall_dot(q)
        J = jacobian(q); Jw = wall_jacobian(q)
        v_tcp = rng.normal(0, 0.03, 3)
        dq_tcp = cartesian_to_joint_vel(q, v_tcp, max_jvel=1.5)
        cart_err = np.linalg.norm(J @ dq_tcp - v_tcp)
        v_wall = rng.normal(0, 0.03, 2)
        dq_wall = wall_velocity_to_joint_vel(q, v_wall, max_jvel=1.5)
        wall_err = np.linalg.norm(Jw @ dq_wall - v_wall) if dot is not None and np.linalg.norm(Jw)>0 else np.nan
        w.writerow({'sample':i, **{f'q{k}':q[k] for k in range(6)},
                    'tcp_x':tcp[0], 'tcp_y':tcp[1], 'tcp_z':tcp[2],
                    'dot_y':dot[0] if dot is not None else np.nan, 'dot_z':dot[1] if dot is not None else np.nan,
                    'on_wall':int(dot is not None), 'cond_tcp':jacobian_condition(J), 'cond_wall':jacobian_condition(Jw),
                    'cart_vel_err':cart_err, 'wall_vel_err':wall_err})
    f.close(); return path


def run_ekf(rng, tag, n=600):
    path = TEST_DIR / f'kalman_{tag}.csv'
    f, w = _writer(path, ['step','true_y','true_z','meas_fk_y','meas_fk_z','meas_cam_y','meas_cam_z','ekf_y','ekf_z','ekf_vy','ekf_vz','sigma_y','sigma_z','innov_y','innov_z','nis','nees','err_m'])
    ekf = LaserDotEKF(dt=0.004, r_fk_std=0.006, r_cam_std=0.018)
    y, z = 0.0, 0.67; vy, vz = 0.08, 0.03
    ekf.reset(y, z)
    for k in range(n):
        if k == 250: vy, vz = -0.06, 0.05
        y += vy * 0.004; z += vz * 0.004
        ekf.predict(None, None)
        meas_fk = np.array([y,z]) + rng.normal(0, 0.006, 2)
        ekf.update_fk(meas_fk[0], meas_fk[1])
        meas_cam = [np.nan, np.nan]
        if k % 16 == 0:
            m = np.array([y,z]) + rng.normal(0, 0.018, 2)
            meas_cam = m.tolist(); ekf.update_camera_absolute(m[0], m[1])
        nees = ekf.nees_against(y,z)
        err = np.linalg.norm(ekf.position - np.array([y,z]))
        w.writerow({'step':k, 'true_y':y, 'true_z':z, 'meas_fk_y':meas_fk[0], 'meas_fk_z':meas_fk[1],
                    'meas_cam_y':meas_cam[0], 'meas_cam_z':meas_cam[1], 'ekf_y':ekf.position[0], 'ekf_z':ekf.position[1],
                    'ekf_vy':ekf.velocity[0], 'ekf_vz':ekf.velocity[1], 'sigma_y':ekf.uncertainty[0], 'sigma_z':ekf.uncertainty[1],
                    'innov_y':ekf.innovation[0], 'innov_z':ekf.innovation[1], 'nis':ekf.nis, 'nees':nees, 'err_m':err})
    f.close(); return path


def run_lqr(rng, tag, n=500):
    path = TEST_DIR / f'lqr_{tag}.csv'
    fields = ['sample','q0','q1','q2','q3','q4','q5','w_tcp','w_wall','cond_wall','gain_mean','gain_max','raw_norm','lqr_norm','null_norm','out_norm','wall_speed_raw','wall_speed_lqr','wall_speed_null']
    f, w = _writer(path, fields)
    for i in range(n):
        q = HOME_POSITIONS + rng.normal(0.0, [0.5,0.4,0.7,0.7,0.7,0.7])
        q[2] = np.clip(q[2], 0.02, np.deg2rad(160))
        raw = rng.uniform(-1, 1, 6)
        Jw = wall_jacobian(q)
        K = lqr_gains(q, yoshikawa(q,'wall'))
        lqr = lqr_velocity_correction(q, raw, q_dot_max=1.0)
        null = null_space_manip_correction(q)
        out = np.clip(lqr + null, -1.0, 1.0)
        w.writerow({'sample':i, **{f'q{k}':q[k] for k in range(6)}, 'w_tcp':yoshikawa(q,'tcp'), 'w_wall':yoshikawa(q,'wall'),
                    'cond_wall':jacobian_condition(Jw), 'gain_mean':np.mean(K), 'gain_max':np.max(K),
                    'raw_norm':np.linalg.norm(raw), 'lqr_norm':np.linalg.norm(lqr), 'null_norm':np.linalg.norm(null), 'out_norm':np.linalg.norm(out),
                    'wall_speed_raw':np.linalg.norm(Jw @ raw), 'wall_speed_lqr':np.linalg.norm(Jw @ lqr), 'wall_speed_null':np.linalg.norm(Jw @ null)})
    f.close(); return path


def run_montecarlo_lines(rng, tag, n=500):
    path = TEST_DIR / f'montecarlo_lines_{tag}.csv'
    f, w = _writer(path, ['sample','length_m','y_min','y_max','z_min','z_max','in_bounds','mean_spacing_m','max_spacing_m'])
    for i in range(n):
        line = random_line(rng=rng)
        spacing = np.linalg.norm(np.diff(line, axis=0), axis=1)
        in_bounds = bool((line[:,0].min()>=WALL_Y_MIN) and (line[:,0].max()<=WALL_Y_MAX) and (line[:,1].min()>=WALL_Z_MIN) and (line[:,1].max()<=WALL_Z_MAX))
        w.writerow({'sample':i, 'length_m':arc_length(line), 'y_min':line[:,0].min(), 'y_max':line[:,0].max(), 'z_min':line[:,1].min(), 'z_max':line[:,1].max(),
                    'in_bounds':int(in_bounds), 'mean_spacing_m':float(np.mean(spacing)), 'max_spacing_m':float(np.max(spacing))})
    f.close(); return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = time.strftime('%Y%m%d_%H%M%S')
    paths = [run_kinematics(rng, tag, args.samples), run_ekf(rng, tag, max(args.samples, 600)),
             run_lqr(rng, tag, args.samples), run_montecarlo_lines(rng, tag, args.samples)]
    print('[component_diagnostics] CSV générés :')
    for p in paths: print(f'  {p}')

    def load_numeric(path):
        import csv
        rows = list(csv.DictReader(open(path, newline='')))
        return rows

    kin, kal, lqr, mc = [load_numeric(p) for p in paths]
    def vals(rows, key):
        out=[]
        for r in rows:
            try:
                v=float(r[key])
                if np.isfinite(v): out.append(v)
            except Exception:
                pass
        return np.asarray(out, dtype=float)

    kin_err=vals(kin,'wall_vel_err')
    kal_err=vals(kal,'err_m')
    raw=vals(lqr,'raw_norm'); out=vals(lqr,'out_norm')
    valid=vals(mc,'in_bounds'); lengths=vals(mc,'length_m')
    checks = [
        ('MGD/MGI', kin_err.size>0 and np.percentile(kin_err,95)<0.02,
         f'erreur wall p95={1000*np.percentile(kin_err,95):.2f} mm/s' if kin_err.size else 'aucune donnée'),
        ('Kalman EKF', kal_err.size>0 and np.percentile(kal_err,95)<0.03,
         f'erreur p95={1000*np.percentile(kal_err,95):.2f} mm' if kal_err.size else 'aucune donnée'),
        ('LQR/singularités', raw.size>0 and out.size>0 and np.all(np.isfinite(out)),
         f'norme moyenne raw={np.mean(raw):.3f} out={np.mean(out):.3f}' if raw.size and out.size else 'aucune donnée'),
        ('Monte-Carlo', valid.size>0 and np.mean(valid)>0.99 and np.min(lengths)>=0.50,
         f'valides={100*np.mean(valid):.1f}% Lmin={np.min(lengths):.2f}m' if valid.size and lengths.size else 'aucune donnée'),
    ]
    print('\n[component_diagnostics] RÉSUMÉ :')
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} {detail}")
    print(f"  Verdict: {sum(int(c[1]) for c in checks)}/{len(checks)} briques PASS")


if __name__ == '__main__':
    main()
