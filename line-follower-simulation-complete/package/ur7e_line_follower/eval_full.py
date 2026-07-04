"""Évaluation post-entraînement + CSV détaillés Monte-Carlo, commande, EKF, RL."""
from __future__ import annotations

import csv
import time
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .env import UR7eLineFollowerEnv
from .train import _load_model_checked
from .reward import DEFAULT_REWARD_PROFILE, REWARD_PROFILES
from .kinematics import fk_ur, jacobian, wall_jacobian, jacobian_condition
from .singularity import yoshikawa, lqr_gains

DATA_DIR = Path.home() / '.ros' / 'ur7e_line_follower'
EVAL_DIR = DATA_DIR / 'eval'
CKPT_DIR = DATA_DIR / 'checkpoints'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS_DEFAULT = [0.0, 0.005, 0.010, 0.020, 0.050]
N_EPISODES_DEFAULT = 50


def _nan_stats(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return dict(mean=np.nan, std=np.nan, p5=np.nan, p50=np.nan, p95=np.nan)
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                p5=float(np.percentile(arr, 5)), p50=float(np.percentile(arr, 50)),
                p95=float(np.percentile(arr, 95)))


def run_eval(model_path: str, n_episodes: int = N_EPISODES_DEFAULT,
             noise_levels: list[float] | None = None, deterministic: bool = True,
             observation_mode: str = 'real', run_tag: str | None = None,
             reward_profile: str | None = None):
    noise_levels = NOISE_LEVELS_DEFAULT if noise_levels is None else noise_levels
    tag = run_tag or time.strftime('%Y%m%d_%H%M%S')
    ep_path = EVAL_DIR / f'eval_episodes_{tag}.csv'
    step_path = EVAL_DIR / f'eval_steps_{tag}.csv'
    mc_path = EVAL_DIR / f'eval_montecarlo_{tag}.csv'
    if reward_profile is None:
        meta_path = Path(str(model_path).removesuffix('.zip') + '.meta.json')
        try:
            reward_profile = json.loads(meta_path.read_text()).get(
                'reward_profile', DEFAULT_REWARD_PROFILE)
        except Exception:
            reward_profile = DEFAULT_REWARD_PROFILE
    if reward_profile not in REWARD_PROFILES:
        reward_profile = DEFAULT_REWARD_PROFILE
    print(f'[eval_full] modèle={model_path}')
    print(f'[eval_full] reward_profile={reward_profile}')
    print(f'[eval_full] sorties={EVAL_DIR}')

    ep_fields = [
        'noise_level_m', 'episode', 'reward', 'success', 'progress', 'n_steps',
        'rmse_m', 'deviation_mean_m', 'deviation_max_m', 'ordered_deviation_mean_m',
        'offwall_ratio', 'yoshikawa_wall_mean', 'cond_wall_median', 'cmd_out_norm_mean',
        'cmd_raw_norm_mean', 'cmd_lqr_ratio_mean', 'null_norm_mean', 'ekf_sigma_mean_m',
        'ekf_nis_mean', 'cam_detection_rate', 'cam_laser_rate',
    ]
    step_fields = [
        'noise_level_m', 'episode', 'step', 'reward', 'done',
        'dot_y', 'dot_z', 'ekf_y', 'ekf_z', 'ekf_vy', 'ekf_vz',
        'ekf_sigma_y', 'ekf_sigma_z', 'ekf_innov_y', 'ekf_innov_z', 'ekf_nis',
        'target_y', 'target_z', 'closest_y', 'closest_z', 'tracking_error_m',
        'progress', 'on_wall', 'offwall_ratio',
        'q0','q1','q2','q3','q4','q5',
        'dq0','dq1','dq2','dq3','dq4','dq5',
        'action_y','action_z',
        'cmd_raw0','cmd_raw1','cmd_raw2','cmd_raw3','cmd_raw4','cmd_raw5',
        'cmd_lqr0','cmd_lqr1','cmd_lqr2','cmd_lqr3','cmd_lqr4','cmd_lqr5',
        'cmd_null0','cmd_null1','cmd_null2','cmd_null3','cmd_null4','cmd_null5',
        'cmd_out0','cmd_out1','cmd_out2','cmd_out3','cmd_out4','cmd_out5',
        'fk_x','fk_y','fk_z','w_tcp','w_wall','cond_tcp','cond_wall',
        'lqr_gain_mean','lqr_gain_max',
        'cam_detected','cam_offset_normal','cam_klt_confidence','cam_cos_t','cam_sin_t','cam_coverage','cam_laser_visible',
    ]
    mc_fields = [
        'noise_level_m', 'episodes', 'reward_mean', 'reward_std', 'reward_p5', 'reward_p50', 'reward_p95',
        'success_rate', 'progress_mean', 'rmse_mean_m', 'rmse_std_m', 'rmse_p5_m', 'rmse_p50_m', 'rmse_p95_m',
        'deviation_mean_m', 'deviation_p95_m', 'offwall_ratio_mean', 'cmd_out_norm_mean', 'ekf_sigma_mean_m',
    ]

    with open(ep_path, 'w', newline='') as ep_f, open(step_path, 'w', newline='') as st_f, open(mc_path, 'w', newline='') as mc_f:
        ep_writer = csv.DictWriter(ep_f, fieldnames=ep_fields); ep_writer.writeheader()
        st_writer = csv.DictWriter(st_f, fieldnames=step_fields); st_writer.writeheader()
        mc_writer = csv.DictWriter(mc_f, fieldnames=mc_fields); mc_writer.writeheader()

        env = UR7eLineFollowerEnv(sensor_noise=False, random_trajectories=True,
                                  update_dot_visual=False,
                                  observation_mode=observation_mode,
                                  reward_profile=reward_profile)
        model = _load_model_checked(model_path, env)
        node = env.node

        for noise in noise_levels:
            env.set_sensor_noise(dot_std_m=float(noise), joint_std_rad=0.002)
            rewards = []; successes = []; progresses = []; rmses = []
            deviations = []; offwalls = []; cmdouts = []; ekfsigmas = []
            print(f'\n[eval_full] bruit spot/caméra σ={noise*1000:.0f} mm')
            for ep in range(1, n_episodes + 1):
                obs, _ = env.reset(); done = False; ep_reward = 0.0; step_i = 0
                ep_rows = []
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated); step_i += 1; ep_reward += float(reward)
                    q = node.joint_pos.copy(); dq = node.joint_vel.copy()
                    dot = node.get_laser_dot()
                    dot_y, dot_z = (float(dot[0]), float(dot[1])) if dot is not None else (np.nan, np.nan)
                    wp_idx = min(env._wp_idx, env._n_wp - 1)
                    tgt_y, tgt_z = map(float, env.waypoints[wp_idx])
                    closest = env._last_closest
                    closest_y, closest_z = map(float, closest.get('closest', [np.nan, np.nan]))
                    track_err = float(closest.get('distance', np.nan))
                    ekf = node.ekf
                    J = jacobian(q); Jw = wall_jacobian(q)
                    gains = lqr_gains(q, yoshikawa(q, 'wall'))
                    # Loguer exactement les 7 valeurs vues par la politique.
                    # En real elles viennent de ROS, en privileged_debug de l'analytique,
                    # en zero elles sont nulles.
                    cam = np.asarray(obs[15:22], dtype=np.float32).copy()
                    row = {
                        'noise_level_m': noise, 'episode': ep, 'step': step_i, 'reward': reward, 'done': int(done),
                        'dot_y': dot_y, 'dot_z': dot_z,
                        'ekf_y': ekf.position[0], 'ekf_z': ekf.position[1], 'ekf_vy': ekf.velocity[0], 'ekf_vz': ekf.velocity[1],
                        'ekf_sigma_y': ekf.uncertainty[0], 'ekf_sigma_z': ekf.uncertainty[1],
                        'ekf_innov_y': ekf.innovation[0], 'ekf_innov_z': ekf.innovation[1], 'ekf_nis': ekf.nis,
                        'target_y': tgt_y, 'target_z': tgt_z, 'closest_y': closest_y, 'closest_z': closest_z,
                        'tracking_error_m': track_err,
                        'progress': info.get('progress', np.nan), 'on_wall': int(info.get('on_wall', False)),
                        'offwall_ratio': info.get('offwall_ratio', np.nan),
                        **{f'q{i}': q[i] for i in range(6)},
                        **{f'dq{i}': dq[i] for i in range(6)},
                        'action_y': float(action[0]), 'action_z': float(action[1]),
                        **{f'cmd_raw{i}': node.last_cmd_raw[i] for i in range(6)},
                        **{f'cmd_lqr{i}': node.last_cmd_lqr[i] for i in range(6)},
                        **{f'cmd_null{i}': node.last_cmd_null[i] for i in range(6)},
                        **{f'cmd_out{i}': node.last_cmd_out[i] for i in range(6)},
                        'fk_x': fk_ur(q)[0], 'fk_y': fk_ur(q)[1], 'fk_z': fk_ur(q)[2],
                        'w_tcp': yoshikawa(q, 'tcp'), 'w_wall': yoshikawa(q, 'wall'),
                        'cond_tcp': jacobian_condition(J), 'cond_wall': jacobian_condition(Jw),
                        'lqr_gain_mean': float(np.mean(gains)), 'lqr_gain_max': float(np.max(gains)),
                        'cam_detected': cam[0], 'cam_offset_normal': cam[1], 'cam_klt_confidence': cam[2],
                        'cam_cos_t': cam[3], 'cam_sin_t': cam[4], 'cam_coverage': cam[5], 'cam_laser_visible': cam[6],
                    }
                    st_writer.writerow(row); ep_rows.append(row)
                ep_f.flush(); st_f.flush()
                rmse = float(info.get('ep_rmse', np.nan))
                ep_summary = {
                    'noise_level_m': noise, 'episode': ep, 'reward': ep_reward,
                    'success': int(info.get('is_success', False)), 'progress': info.get('progress', np.nan), 'n_steps': step_i,
                    'rmse_m': rmse, 'deviation_mean_m': info.get('deviation_mean', np.nan),
                    'deviation_max_m': info.get('deviation_max', np.nan),
                    'ordered_deviation_mean_m': info.get('ordered_deviation_mean', np.nan),
                    'offwall_ratio': info.get('offwall_ratio', np.nan),
                    'yoshikawa_wall_mean': _nan_stats([r['w_wall'] for r in ep_rows])['mean'],
                    'cond_wall_median': _nan_stats([r['cond_wall'] for r in ep_rows])['p50'],
                    'cmd_out_norm_mean': _nan_stats([np.linalg.norm([r[f'cmd_out{i}'] for i in range(6)]) for r in ep_rows])['mean'],
                    'cmd_raw_norm_mean': _nan_stats([np.linalg.norm([r[f'cmd_raw{i}'] for i in range(6)]) for r in ep_rows])['mean'],
                    'cmd_lqr_ratio_mean': _nan_stats([
                        np.linalg.norm([r[f'cmd_lqr{i}'] for i in range(6)]) / max(np.linalg.norm([r[f'cmd_raw{i}'] for i in range(6)]), 1e-9)
                        for r in ep_rows])['mean'],
                    'null_norm_mean': _nan_stats([np.linalg.norm([r[f'cmd_null{i}'] for i in range(6)]) for r in ep_rows])['mean'],
                    'ekf_sigma_mean_m': _nan_stats([(r['ekf_sigma_y'] + r['ekf_sigma_z']) / 2 for r in ep_rows])['mean'],
                    'ekf_nis_mean': _nan_stats([r['ekf_nis'] for r in ep_rows])['mean'],
                    'cam_detection_rate': _nan_stats([r['cam_detected'] for r in ep_rows])['mean'],
                    'cam_laser_rate': _nan_stats([r['cam_laser_visible'] for r in ep_rows])['mean'],
                }  # fin ep_summary
                ep_writer.writerow(ep_summary); ep_f.flush()
                rewards.append(ep_reward); successes.append(ep_summary['success']); progresses.append(ep_summary['progress'])
                rmses.append(rmse); deviations.append(ep_summary['deviation_mean_m']); offwalls.append(ep_summary['offwall_ratio'])
                cmdouts.append(ep_summary['cmd_out_norm_mean']); ekfsigmas.append(ep_summary['ekf_sigma_mean_m'])
                print(f"  ep {ep:03d} rew={ep_reward:+.1f} ok={ep_summary['success']} "
                      f"prog={ep_summary['progress']*100:.0f}% rmse={rmse*100:.1f}cm")
            rstat = _nan_stats(rewards); rmstat = _nan_stats(rmses); devstat = _nan_stats(deviations)
            mc_writer.writerow({
                'noise_level_m': noise, 'episodes': n_episodes,
                'reward_mean': rstat['mean'], 'reward_std': rstat['std'], 'reward_p5': rstat['p5'],
                'reward_p50': rstat['p50'], 'reward_p95': rstat['p95'],
                'success_rate': float(np.mean(successes)), 'progress_mean': _nan_stats(progresses)['mean'],
                'rmse_mean_m': rmstat['mean'], 'rmse_std_m': rmstat['std'], 'rmse_p5_m': rmstat['p5'],
                'rmse_p50_m': rmstat['p50'], 'rmse_p95_m': rmstat['p95'],
                'deviation_mean_m': devstat['mean'], 'deviation_p95_m': devstat['p95'],
                'offwall_ratio_mean': _nan_stats(offwalls)['mean'],
                'cmd_out_norm_mean': _nan_stats(cmdouts)['mean'],
                'ekf_sigma_mean_m': _nan_stats(ekfsigmas)['mean'],
            })
            mc_f.flush()
        env.close()
    print(f'\n[eval_full] écrit :\n  {ep_path}\n  {step_path}\n  {mc_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='')
    parser.add_argument('--episodes', type=int, default=N_EPISODES_DEFAULT)
    parser.add_argument('--noise-levels', nargs='+', type=float, default=NOISE_LEVELS_DEFAULT)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--reward-profile', choices=REWARD_PROFILES, default=None,
                        help='défaut: profil lu dans le .meta.json du modèle')
    args = parser.parse_args()
    model_path = args.model
    if not model_path:
        zips = sorted(CKPT_DIR.glob('*.zip'))
        if not zips:
            raise FileNotFoundError(f'Aucun checkpoint dans {CKPT_DIR}')
        model_path = str(zips[-1])
    base_tag = time.strftime('%Y%m%d_%H%M%S')
    campaigns = [
        ('real',            'real'),
        ('privileged_debug', 'privileged_debug'),
        ('zero_camera',     'zero'),
    ]
    for mode_tag, obs_mode in campaigns:
        print(f'\n[eval_full] === Campagne {mode_tag} ===')
        run_eval(model_path, args.episodes, args.noise_levels,
                 deterministic=not args.stochastic,
                 observation_mode=obs_mode,
                 run_tag=f'{base_tag}_{mode_tag}',
                 reward_profile=args.reward_profile)


if __name__ == '__main__':
    main()
