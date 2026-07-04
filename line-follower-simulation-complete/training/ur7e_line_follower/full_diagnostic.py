"""Diagnostic intégré avec sorties lisibles pour chaque brique du système."""
from __future__ import annotations
import argparse
import math
import numpy as np
import cv2
import rclpy

from .bridge import HOME_POSITIONS
from .kinematics import fk_ur, laser_wall_dot, wall_jacobian, wall_velocity_to_joint_vel, jacobian_condition
from .ekf import LaserDotEKF
from .singularity import yoshikawa, command_filter_diagnostics, check_known_singularities
from .target_line import (random_line_from_start, DEFAULT_HOME_DOT, arc_length,
                          WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX)
from .camera_line_detector import CameraLineDetector
from .expert_follow import run as run_expert


def mark(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
    return bool(ok)


def offline(seed=42, samples=200):
    print('\n========== DIAGNOSTICS HORS LIGNE ==========')
    rng = np.random.default_rng(seed)
    checks = []

    tcp = fk_ur(HOME_POSITIONS)
    dot = laser_wall_dot(HOME_POSITIONS)
    checks.append(mark('MGD / FK', tcp.shape == (3,) and np.all(np.isfinite(tcp)), f'TCP={np.round(tcp,4)}'))
    checks.append(mark('MGD / laser mur', dot is not None and np.all(np.isfinite(dot)), f'dot={None if dot is None else np.round(dot,4)}'))

    Jw = wall_jacobian(HOME_POSITIONS)
    errs = []
    for v in ([0.03,0.0],[-0.03,0.0],[0.0,0.03],[0.0,-0.03]):
        v = np.asarray(v, dtype=float)
        dq = wall_velocity_to_joint_vel(HOME_POSITIONS, v, max_jvel=0.35)
        errs.append(float(np.linalg.norm(Jw @ dq - v)))
    checks.append(mark('MGI différentielle', max(errs) < 0.015, f'erreur max={1000*max(errs):.2f} mm/s cond={jacobian_condition(Jw):.2f}'))

    ekf = LaserDotEKF(dt=0.004)
    ekf.reset(0.0, 0.48)
    target = np.array([0.12,0.72])
    for _ in range(100):
        ekf.predict(None, None)
        m = target + rng.normal(0, 0.004, 2)
        ekf.update_fk(m[0], m[1])
    e = float(np.linalg.norm(ekf.position-target))
    checks.append(mark('Kalman EKF', e < 0.01, f'erreur={1000*e:.2f} mm sigma={1000*np.mean(ekf.uncertainty):.2f} mm'))

    raw = np.array([0.30,-0.20,0.10,0.15,-0.12,0.05])
    diag = command_filter_diagnostics(HOME_POSITIONS, raw, q_dot_max=0.35)
    lqr_ok = np.all(np.isfinite(diag['out_cmd'])) and diag['out_cmd_norm'] > 0 and diag['out_cmd_norm'] <= np.linalg.norm(raw)+0.2
    checks.append(mark('LQR + espace nul', lqr_ok, f"raw={diag['raw_cmd_norm']:.3f} out={diag['out_cmd_norm']:.3f} null={diag['null_cmd_norm']:.3f}"))

    q_sing = HOME_POSITIONS.copy(); q_sing[2] = 0.03
    w_home = yoshikawa(HOME_POSITIONS, 'wall'); w_sing = yoshikawa(q_sing, 'wall')
    flags = check_known_singularities(q_sing)
    checks.append(mark('Singularités', w_sing < w_home and flags['elbow']['is_near'], f'w_home={w_home:.5f} w_sing={w_sing:.5f}'))

    valid = 0; lengths=[]; start_err=[]
    for _ in range(samples):
        line = random_line_from_start(rng, DEFAULT_HOME_DOT)
        lengths.append(arc_length(line)); start_err.append(np.linalg.norm(line[0]-DEFAULT_HOME_DOT))
        inside = (line[:,0].min() >= WALL_Y_MIN and line[:,0].max() <= WALL_Y_MAX and
                  line[:,1].min() >= WALL_Z_MIN and line[:,1].max() <= WALL_Z_MAX)
        valid += int(inside and lengths[-1] >= 0.50)
    checks.append(mark('Monte-Carlo dessins', valid == samples and max(start_err) < 1e-5, f'{valid}/{samples} valides Lmin={min(lengths):.2f}m départ_err={1000*max(start_err):.3f}mm'))

    detector = CameraLineDetector.__new__(CameraLineDetector)
    detector._init_state()
    frame = np.full((480,640,3), 245, dtype=np.uint8)
    pts = np.array([[110,390],[180,340],[260,300],[340,230],[430,180],[530,100]], np.int32)
    cv2.polylines(frame,[pts],False,(255,80,20),12)
    cv2.circle(frame,(110,390),10,(0,0,255),-1)
    results=[]
    for _ in range(5): results.append(detector.process_frame(frame))
    v=np.asarray(results[-1].detection_vector)
    g=np.asarray(results[-1].guidance_vector)
    checks.append(mark('KLT + lookahead synth.', v[0] > .5 and v[2] > 0 and v[6] > .5 and np.linalg.norm(g[:2]) > 1e-3, f'detection={np.round(v,3)} guidance={np.round(g,3)}'))

    ok = all(checks)
    print(f"\n[offline] {'TOUT PASS' if ok else 'ÉCHEC À CORRIGER'} : {sum(checks)}/{len(checks)} briques")
    return ok


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-only', action='store_true')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()
    if not rclpy.ok(): rclpy.init()
    try:
        ok=offline(args.seed,args.samples)
        if not args.offline_only:
            print('\n========== DIAGNOSTIC INTÉGRÉ GAZEBO ==========')
            try:
                result = run_expert(steps=args.steps)
                live_ok=(result['line_rate'] >= .80 and result['laser_rate'] >= .80 and
                         result.get('lookahead_valid_rate', 0.0) >= .70 and
                         result['progress'] >= .50 and result['cond_max'] < 1000)
                mark('Chaîne intégrée', live_ok,
                     f"prog={100*result['progress']:.1f}% ligne={100*result['line_rate']:.1f}% laser={100*result['laser_rate']:.1f}% lookahead={100*result.get('lookahead_valid_rate',0):.1f}% arrêt={result.get('stop_reason','n/a')}")
            except Exception as exc:
                live_ok = False
                mark('Chaîne intégrée', False, f'{type(exc).__name__}: {exc}')
            ok = ok and live_ok
        print(f"\n========== VERDICT GLOBAL : {'PASS' if ok else 'FAIL'} ==========")
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__': main()
