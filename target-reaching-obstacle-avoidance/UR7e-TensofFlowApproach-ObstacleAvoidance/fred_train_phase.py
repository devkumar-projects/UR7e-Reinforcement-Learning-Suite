# ==============================================================================
# FICHIER : fred_train_phase.py
# RÔLE : Entraînement SAC (tf-agents) générique pour une phase, avec TRANSFER
#        LEARNING : peut reprendre les poids d'une phase précédente.
#        Deux trackers : glissant stochastique (continu) + eval déterministe
#        lissé (tous les 1000 pas). Métriques -> CSV par phase.
#
# Appelé par fred_train_phase1.py / _phase2.py / _phase3.py.
# ==============================================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.utils import common

from fred_metrics import (MetricsTracker, SUCCESS_REWARD_THRESHOLD,
                          COLLISION_REWARD_THRESHOLD)

LEARNING_RATE = 3e-4
BATCH_SIZE = 256
REPLAY_CAPACITY = 200_000
INITIAL_COLLECT = 1_000
LOG_INTERVAL = 1_000
EVAL_INTERVAL = 1_000           # eval déterministe tous les 1000 pas
NUM_EVAL_EPISODES = 10
CSV_INTERVAL = 1_000            # snapshot CSV tous les 1000 pas
CHECKPOINT_INTERVAL = 2_000
ACTOR_FC = (256, 256)
CRITIC_FC = (256, 256)


def _proj_net(action_spec):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec, mean_transform=None, state_dependent_std=True,
        init_means_output_factor=0.1,
        std_transform=sac_agent.std_clip_transform, scale_distribution=True)


def build_agent(train_env):
    obs_spec = train_env.observation_spec()
    act_spec = train_env.action_spec()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        obs_spec, act_spec, fc_layer_params=ACTOR_FC,
        continuous_projection_net=_proj_net)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, act_spec), joint_fc_layer_params=CRITIC_FC)
    agent = sac_agent.SacAgent(
        train_env.time_step_spec(), act_spec,
        actor_network=actor_net, critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        critic_optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        alpha_optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        target_update_tau=0.005, target_update_period=1,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99, reward_scale_factor=1.0,
        train_step_counter=tf.Variable(0))
    agent.initialize()
    return agent


def evaluate_success(env, policy, n_episodes):
    succ = 0
    for _ in range(n_episodes):
        t = env.reset()
        last_r = 0.0
        while not t.is_last():
            a = policy.action(t)
            t = env.step(a.action)
            last_r = float(t.reward.numpy()[0])
        if last_r >= SUCCESS_REWARD_THRESHOLD:
            succ += 1
    return succ / n_episodes


def train_phase(make_env, num_iterations, csv_path, policy_save_dir,
                load_from=None, seed=0):
    py_env = make_env(seed)
    eval_py_env = make_env(seed + 123)
    train_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = build_agent(train_env)
    agent.train = common.function(agent.train)

    replay = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size, max_length=REPLAY_CAPACITY)

    # --- TRANSFER LEARNING via Checkpointer (mécanisme officiel tf-agents) ---
    # Le Checkpointer sauvegarde/restaure les VARIABLES des réseaux actor+critic
    # +alpha. C'est ce qui permet de reprendre les poids d'une phase à l'autre,
    # contrairement au PolicySaver (qui sert l'inférence, pas le transfer).
    ckpt_dir = policy_save_dir + "_ckpt"
    train_checkpointer = common.Checkpointer(
        ckpt_dir=ckpt_dir, max_to_keep=2, agent=agent,
        policy=agent.policy, global_step=agent.train_step_counter)

    if load_from is not None:
        load_ckpt = load_from + "_ckpt"
        if os.path.isdir(load_ckpt):
            print(f"[transfer] reprise des poids depuis '{load_ckpt}'")
            transfer_ckpt = common.Checkpointer(
                ckpt_dir=load_ckpt, max_to_keep=2, agent=agent,
                policy=agent.policy, global_step=agent.train_step_counter)
            transfer_ckpt.initialize_or_restore()
            # on remet le compteur de pas à zéro pour la nouvelle phase
            agent.train_step_counter.assign(0)
            print("[transfer] poids actor+critic restaurés ; compteur remis à 0.")
        else:
            print(f"[transfer] '{load_ckpt}' introuvable -> démarrage à froid.")

    metrics = MetricsTracker(csv_path)

    # tracker glissant continu via observateur de fin d'épisode (eager)
    def episode_observer(traj):
        is_last = np.atleast_1d(traj.is_last().numpy())
        rew = np.atleast_1d(traj.reward.numpy())
        for k in range(is_last.shape[0]):
            metrics.record_step(rew[k])
            if bool(is_last[k]):
                # distance finale approx via l'env (dernière mesurée)
                dist_cm = py_env._last_distance_cm
                metrics.end_episode(py_env._last_terminal_reward, dist_cm)

    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())
    dynamic_step_driver.DynamicStepDriver(
        train_env, random_policy, observers=[replay.add_batch],
        num_steps=INITIAL_COLLECT).run()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env, agent.collect_policy,
        observers=[replay.add_batch, episode_observer], num_steps=1)

    dataset = replay.as_dataset(num_parallel_calls=3,
                                sample_batch_size=BATCH_SIZE,
                                num_steps=2).prefetch(3)
    iterator = iter(dataset)
    saver = policy_saver.PolicySaver(agent.policy)

    pbar = tqdm(range(num_iterations), desc="SAC", unit="it", dynamic_ncols=True)
    try:
        for _ in pbar:
            collect_driver.run()
            exp, _ = next(iterator)
            loss = float(agent.train(exp).loss)
            step = int(agent.train_step_counter.numpy())

            if step % EVAL_INTERVAL == 0:
                ev = evaluate_success(eval_env, agent.policy, NUM_EVAL_EPISODES)
                metrics.update_eval(ev)

            pbar.set_postfix({
                "succ": f"{metrics.success_rate:.2f}",
                "coll": f"{metrics.collision_rate:.2f}",
                "to": f"{metrics.timeout_rate:.2f}",
                "ev_succ": f"{metrics.eval_success_rate:.2f}",
                "d_cm": f"{metrics.mean_distance_cm:.1f}",
                "loss": f"{loss:.2f}",
            })

            if step % LOG_INTERVAL == 0:
                tqdm.write(f"[step {step:>7d}] succ={metrics.success_rate:.2f} "
                           f"coll={metrics.collision_rate:.2f} "
                           f"to={metrics.timeout_rate:.2f} "
                           f"ev_succ={metrics.eval_success_rate:.2f} "
                           f"d={metrics.mean_distance_cm:.1f}cm ep={metrics.episodes}")
            if step % CSV_INTERVAL == 0:
                metrics.snapshot(step)
            if step % CHECKPOINT_INTERVAL == 0:
                saver.save(policy_save_dir)        # pour la visualisation
                train_checkpointer.save(step)      # pour le transfer
    except KeyboardInterrupt:
        tqdm.write("\n[Ctrl+C] sauvegarde...")
        saver.save(policy_save_dir + "_interrupted")
        train_checkpointer.save(int(agent.train_step_counter.numpy()))
        metrics.snapshot(int(agent.train_step_counter.numpy()))
    else:
        saver.save(policy_save_dir + "_final")
        train_checkpointer.save(int(agent.train_step_counter.numpy()))
        metrics.snapshot(int(agent.train_step_counter.numpy()))
    finally:
        py_env.close(); eval_py_env.close()
    print(f"Terminé. Politique -> '{policy_save_dir}_final', "
          f"checkpoint -> '{ckpt_dir}', métriques -> '{csv_path}'")
