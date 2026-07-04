"""Baseline géométrique: suit le dessin courant sans modèle RL.

Ce n'est pas le contrôleur final. Il sert à isoler la chaîne caméra/KLT,
MGD/MGI, LQR, Kalman et singularités avant l'entraînement SAC.
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import rclpy

from .env import UR7eLineFollowerEnv
from .control import MAX_WALL_SPEED_M_S
from .target_line import closest_point_on_polyline


def _stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 'n/a'
    return f'moy={a.mean():.4f} min={a.min():.4f} max={a.max():.4f}'


def run(steps: int = 500, kp: float = 2.2, trials_per_drawing: int = 5,
        profile: str = 'realistic') -> dict:
    minimal = profile == 'minimal_straight_line_debug'
    env = UR7eLineFollowerEnv(
        random_trajectories=not minimal,
        sensor_noise=False,
        observation_mode='privileged_debug' if minimal else 'real',
        update_dot_visual=False,
        trials_per_trajectory=trials_per_drawing,
        curriculum=not minimal,
        training_profile=profile,
        deterministic_pulse=True,
        guided_reset=True,
    )
    rows = []
    stop_reason = 'steps_exhausted'
    try:
        obs, _ = env.reset()
        print(f'[expert] dessin: {env._n_wp} points | longueur={env._path_length:.3f} m')
        print(f'[expert] départ={np.round(env.waypoints[0],4)} arrivée={np.round(env.waypoints[-1],4)}')
        for k in range(int(steps)):
            dot = env.node.get_laser_dot()
            target_idx = min(env._wp_idx, env._n_wp - 1)
            target = np.asarray(env.waypoints[target_idx], dtype=float)
            if dot is None:
                action = np.zeros(2, dtype=np.float32)
            else:
                err = target - np.asarray(dot, dtype=float)
                desired = kp * err
                speed = float(np.linalg.norm(desired))
                if speed > MAX_WALL_SPEED_M_S:
                    desired *= MAX_WALL_SPEED_M_S / speed
                action = np.clip(desired / MAX_WALL_SPEED_M_S, -1.0, 1.0).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            guidance = np.asarray(obs[12:15], dtype=float)
            cam = np.asarray(obs[15:22], dtype=float)
            dot2 = env.node.get_laser_dot()
            true_d = math.nan
            signed = math.nan
            ekf_err = math.nan
            if dot2 is not None:
                dot2 = np.asarray(dot2, dtype=float)
                c = closest_point_on_polyline(dot2, env.waypoints,
                                              start_idx=max(env._wp_idx - 1, 0), window=5)
                true_d = float(c['distance'])
                seg = min(int(c['segment_index']), len(env.waypoints)-2)
                t = np.asarray(env.waypoints[seg+1]-env.waypoints[seg], dtype=float)
                t /= max(float(np.linalg.norm(t)), 1e-12)
                n = np.array([-t[1], t[0]])
                signed = float(np.dot(dot2 - np.asarray(c['closest']), n))
                ekf_err = float(np.linalg.norm(env.node.ekf.position - dot2))
            row = {
                'step': k,
                'progress': float(info.get('progress', 0.0)),
                'distance': true_d,
                'signed': signed,
                'detected': cam[0],
                'offset': cam[1],
                'confidence': cam[2],
                'tangent_norm': float(math.hypot(cam[3], cam[4])),
                'coverage': cam[5],
                'lookahead_u': guidance[0],
                'lookahead_v': guidance[1],
                'visual_progress': guidance[2],
                'laser': cam[6],
                'ekf_err': ekf_err,
                'cond': float(info.get('cond_wall', math.nan)),
                'w': float(info.get('yoshikawa_w', math.nan)),
                'cmd_raw': float(info.get('cmd_raw_norm', math.nan)),
                'cmd_out': float(info.get('cmd_out_norm', math.nan)),
                'reward': float(reward),
            }
            rows.append(row)
            if k % 10 == 0 or terminated or truncated:
                print(
                    f"[expert] step={k:03d} wp={env._wp_idx:02d}/{env._n_wp} "
                    f"prog={100*row['progress']:5.1f}% d={100*true_d:6.2f}cm "
                    f"KLT={cam[0]:.0f}/{cam[1]:+.3f}/{cam[2]:.3f}/{cam[5]:.3f}/{cam[6]:.0f} "
                    f"LA={guidance[0]:+.2f}/{guidance[1]:+.2f}/{guidance[2]:.2f} "
                    f"cond={row['cond']:.1f} w={row['w']:.4f}"
                )
            if terminated or truncated:
                timeout_keys = [key for key, value in info.items()
                                if key.endswith('_timeout') and bool(value)]
                if terminated:
                    if bool(info.get('is_success', False)):
                        stop_reason = 'trajectory_completed'
                    elif bool(info.get('stagnation_timeout', False)):
                        stop_reason = 'stagnation_timeout'
                    elif bool(info.get('offwall_timeout', False)):
                        stop_reason = 'offwall_timeout'
                    else:
                        stop_reason = 'terminated_unspecified'
                elif timeout_keys:
                    stop_reason = ','.join(timeout_keys)
                elif truncated:
                    stop_reason = 'max_steps_or_external_truncation'
                print(f'[expert] fin: terminated={terminated} truncated={truncated} reason={stop_reason}')
                break

        arr = lambda name: np.asarray([r[name] for r in rows], dtype=float)
        mask = (arr('detected') > 0.5) & (arr('laser') > 0.5) & np.isfinite(arr('signed'))
        corr = math.nan
        if np.count_nonzero(mask) >= 10 and np.std(arr('offset')[mask]) > 1e-8 and np.std(arr('signed')[mask]) > 1e-8:
            corr = float(np.corrcoef(arr('offset')[mask], arr('signed')[mask])[0,1])
        signed_std = float(np.nanstd(arr('signed')[mask])) if np.count_nonzero(mask) else math.nan
        offset_std = float(np.nanstd(arr('offset')[mask])) if np.count_nonzero(mask) else math.nan
        corr_excited = bool(
            np.isfinite(signed_std) and np.isfinite(offset_std)
            and signed_std >= 0.003 and offset_std >= 0.005
        )
        result = {
            'steps': len(rows),
            'progress': float(rows[-1]['progress']) if rows else 0.0,
            'line_rate': float(np.mean(arr('detected'))) if rows else 0.0,
            'laser_rate': float(np.mean(arr('laser'))) if rows else 0.0,
            'klt_conf_mean': float(np.nanmean(arr('confidence'))) if rows else 0.0,
            'lookahead_valid_rate': float(np.mean(np.hypot(arr('lookahead_u'), arr('lookahead_v')) > 1e-3)) if rows else 0.0,
            'visual_progress_final': float(rows[-1]['visual_progress']) if rows else 0.0,
            'distance_mean': float(np.nanmean(arr('distance'))) if rows else math.nan,
            'ekf_error_mean': float(np.nanmean(arr('ekf_err'))) if rows else math.nan,
            'corr_klt_metric': corr,
            'corr_excited': corr_excited,
            'signed_std': signed_std,
            'offset_std': offset_std,
            'cond_max': float(np.nanmax(arr('cond'))) if rows else math.nan,
            'yoshikawa_min': float(np.nanmin(arr('w'))) if rows else math.nan,
            'stop_reason': stop_reason,
        }
        print('\n========== BILAN SUIVI EXPERT ==========')
        print(f"progression       : {100*result['progress']:.1f}%")
        print(f"détection ligne   : {100*result['line_rate']:.1f}%")
        print(f"détection laser   : {100*result['laser_rate']:.1f}%")
        print(f"confiance KLT     : {result['klt_conf_mean']:.3f}")
        print(f"lookahead valide  : {100*result['lookahead_valid_rate']:.1f}%")
        print(f"progression image : {100*result['visual_progress_final']:.1f}%")
        print(f"distance moyenne  : {100*result['distance_mean']:.2f} cm")
        print(f"erreur EKF moyenne: {1000*result['ekf_error_mean']:.2f} mm")
        print(f"corr KLT/métrique : {result['corr_klt_metric']:+.3f}")
        print(
            f"excitation latérale: sigma_metric={1000*result['signed_std']:.2f} mm "
            f"sigma_KLT={result['offset_std']:.4f}"
        )
        if not result['corr_excited']:
            print('[INFO] Corrélation non concluante : le contrôleur expert reste trop près de la ligne.')
        print(f"cond(Jwall) max   : {result['cond_max']:.2f}")
        print(f"Yoshikawa min     : {result['yoshikawa_min']:.5f}")
        print(f"arrêt diagnostic  : {result['stop_reason']}")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--kp', type=float, default=2.2)
    parser.add_argument('--trials-per-drawing', type=int, default=5)
    parser.add_argument('--profile', default='realistic',
                        choices=['realistic', 'minimal_straight_line_debug'])
    args = parser.parse_args()
    if not rclpy.ok():
        rclpy.init()
    try:
        run(args.steps, args.kp, args.trials_per_drawing, args.profile)
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
