from agents import q_agent
from analysis.low_rank import (rank, tabular_q_matrix, hankel_policy,
                               autoregressive_value_probe)
from analysis.registry import resolve_methods
import gymnasium as gym
import numpy as np
from tqdm import tqdm


def run_analysis_tick(agent, env, analysis_config: dict, run_logger=None, episode=None):
    """Rank + Hankel analysis dispatched every ep_freq episodes/iterations.
    The Hankel sweep rolls `env` out to termination — callers must reset it after."""
    for method, names in resolve_methods(analysis_config.get("methods")):
        results = method(agent=agent, env=env)
        if not isinstance(results, tuple): # Deals with functions returning multiple matrices for analysis
            results = (results,)
        for matrix, name in zip(results, names):
            print(f"****************************{name}****************************")
            # With a run_logger the spectrum figure goes to disk instead of inline —
            # 15k-episode runs would otherwise embed hundreds of figures in the notebook.
            save_to = run_logger.figure_path(name, episode) if run_logger is not None else None
            r, sr, spk, shape, irs, ics, rc, cc, nzr, nzc = rank.row_rank_property_check(
                matrix, name, save_to=save_to, show=run_logger is None)
            # returns effective_rank, stable_rank, spikiness, shape, irs/ics (normalised top-r leverage
            # per row/col), rc/cc = coherence of the rank-r row/col space ((dim/rank)*max leverage, in
            # [1, dim/rank]), nzr and nzc are the number of non zero rows and columns in the original matrix.
            print(f"eff_rank: {r}, stable_rank: {sr:.2f}, spikiness: {spk:.2f}, shape: {shape}, non-zero rows :{nzr}, non-zero cols:{nzc}")
            print(f"top-r leverage spread: row min={irs.min():.4g} max={irs.max():.4g} (uniform {1.0/shape[0]:.4g}) | col min={ics.min():.4g} max={ics.max():.4g} (uniform {1.0/shape[1]:.4g})")
            print(f"coherence score: row={rc:.4g} col={cc:.4g}")
            if run_logger is not None:
                run_logger.log_rank_stats(episode, name, r, sr, spk, shape,
                                          irs, ics, rc, cc, nzr, nzc)
    # Richer Hankel analysis: several rollouts + growing sub-trajectories,
    # self-logging to hankel_sweep.csv / trajectories/ / figures. Opt-in via
    # the analysis.hankel_sweep config block (absent => skipped entirely).
    hk_cfg = analysis_config.get("hankel_sweep")
    if hk_cfg and hk_cfg.get("enabled", True):
        hankel_policy.hankel_sweep_analysis(agent, env, hk_cfg,
                                            run_logger=run_logger, episode=episode)
    # Autoregressive value-recurrence probe: freeze the policy, roll greedy
    # trajectories, fit one linear recurrence per order and record how well it
    # predicts held-out value sequences. Opt-in via the
    # analysis.autoregressive_value_probe config block (absent => skipped).
    # Like the Hankel sweep it rolls `env` to termination, so the same
    # caller-resets-afterwards contract applies.
    ar_cfg = analysis_config.get("autoregressive_value_probe")
    if ar_cfg and ar_cfg.get("enabled", True):
        probe_kwargs = {k: v for k, v in ar_cfg.items() if k != "enabled"}
        if "orders" in probe_kwargs:
            probe_kwargs["orders"] = tuple(probe_kwargs["orders"])
        summary = autoregressive_value_probe.autoregressive_value_probe(
            agent, env, run_logger=run_logger, episode=episode, **probe_kwargs)
        _print_autoregressive_summary(summary)


def _print_autoregressive_summary(summary):
    """One compact line per recurrence order, for the training-run console."""
    results = summary["results"]
    if not results:
        print("autoregressive value probe: no trajectory long enough to fit")
        return
    lengths = [len(s) for s in summary["value_sequences"]]
    print(f"**************** autoregressive value probe "
          f"({len(lengths)} trajectories, mean length {np.mean(lengths):.0f}) "
          f"****************")
    for order, per_split in sorted(results.items()):
        held_out = per_split.get(
            autoregressive_value_probe.HELD_OUT_TRAJECTORY_SPLIT)
        if held_out is None:
            continue
        test = held_out["test"]
        per_window = "  ".join(
            f"tau={horizon} {metrics['one_minus_r_squared']:.4f}"
            + ("!" if metrics["diverged"] else "")
            for horizon, metrics in sorted(held_out["test_horizons"].items()))
        print(f"  order {order}: held-out test 1 - R^2  "
              f"one-step {test['one_minus_r_squared_one_step_ahead']:.4f} | "
              f"by forecast window: {per_window}")


def dqn_training_loop(agent: q_agent.QAgent, env: gym.Env,
                      no_episodes: int, target_network_update_steps: int,
                      train_frequency_steps: int, use_episode_training: bool, solved_reward: int,
                      warmup_steps: int = 0, early_stopping_patience_eps: int = 50,
                      np_seed: int = 52, no_eps_to_avg: int = 10,
                      analysis_config: dict | None = None,
                      DEBUG=False, atari= False, run_logger=None):
    """The default behaviour is that we wait for atleast train_frequency_steps between training. However this is only invoked after every
    episode. This means that if after train_frequency_steps it will only update once the episode ends and not in between.

    Args:
        agent (q_agent.QAgent): _description_
        env (gym.Env): _description_
        no_episodes (int): _description_
        target_network_update_steps (int): _description_
        train_frequency_steps (int): _description_
        use_episode_training (int): _description_
        warmup_steps (int, optional): _description_. Defaults to 0.
        np_seed (int, optional): _description_. Defaults to 52.
        no_eps_to_avg (int, optional): _description_. Defaults to 10.
        analysis_config (dict, optional): _description_. Defaults to None (== {}, no analysis).
        DEBUG (bool, optional): _description_. Defaults to False.
        run_logger (RunLogger, optional): when provided, analysis figures are
            saved to its run directory instead of rendered inline, rank stats go
            to rank_stats.csv, rewards to rewards.csv, and checkpoints are kept
            (latest at every analysis tick, best on new reward-window high, final
            on return). Defaults to None (old behaviour, everything inline).

    Returns:
        _type_: _description_
    """
    analysis_config = analysis_config or {}
    state, info = env.reset(seed = np_seed)
    s_tn_upd, s_train, step_count = 0,0,0
    episode_rewards_training = []
    episode_steps_training = []  # env steps per episode -> rewards.csv steps column
    best_window_avg = float("-inf")
    for episode in tqdm(range(no_episodes)):
        # Do a policy rollout and explore with the current agent
        epsiode_total_reward = 0
        episode_steps = 0
        terminated = truncated = False
        # This is one rollout from the current policy pi = DQN(theeta) 
        # which is updated once every episode
        while not (terminated or truncated):
            action = agent.pi(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_buffer(state, action, reward, next_state, terminated, truncated) if not atari else  agent.update_buffer_atari(state, action, reward, next_state, terminated, truncated)
            state = next_state
            # A ScaleReward-wrapped env trains the agent on scaled rewards but exposes
            # the raw game score in info — log raw so reward curves and solved_reward
            # stay comparable across runs with and without scaling.
            epsiode_total_reward += info.get("raw_reward", reward)
            episode_steps += 1

            if step_count > warmup_steps: # start the counters for training and updating target network
                # This is used to bring the Network used to Calculate Q-Targets up to date with the policy network
                agent.decay_epsilon()
                s_tn_upd+= 1 
                s_train+= 1
            step_count += 1
            
            # Check if enough steps have passed to update the target network
            if s_tn_upd >= target_network_update_steps:
                s_tn_upd = 0
                agent.update_target_network()
            
            if s_train >= train_frequency_steps and not use_episode_training:
                s_train = 0
                # train the agent
                diag = agent.train() # train every train_frequency_steps steps
                if diag is not None and run_logger is not None:
                    run_logger.log_train_diagnostics(episode, **diag)

        # reset the environment for next episode
        state, _ = env.reset()
        episode_rewards_training.append(epsiode_total_reward)
        episode_steps_training.append(episode_steps)

        # We will do the training only when the episode had ended... and ensure that at least train_frequency_steps have passed since the last training
        # This is mainly relevant only when use_episode_training is True
        if s_train >= train_frequency_steps:
            s_train = 0
            # train the agent
            diag = agent.train() # train every train_frequency_steps steps
            if diag is not None and run_logger is not None:
                run_logger.log_train_diagnostics(episode, **diag)

        # Optional per-episode agent hook — e.g. FHR-DQN's automatic lambda
        # ramp-down watches episode rewards against its own residual trend.
        # After the post-episode train() block, so that under
        # use_episode_training the episode's own gradient steps (and their
        # penalty residuals) are attributed to this episode, not the next.
        episode_hook = getattr(agent, "notify_episode_end", None)
        if episode_hook is not None:
            episode_hook(episode, epsiode_total_reward)
        
        # print training status
        if episode % no_eps_to_avg == 0:
            window = episode_rewards_training[-no_eps_to_avg:]
            ep_avg = sum(window) / len(window)
            print(f"episode {episode} avg_rewarg: {ep_avg}")
            if run_logger is not None and ep_avg > best_window_avg:
                best_window_avg = ep_avg
                run_logger.checkpoint(agent, "best")
        
        # print more detailed training status 
        # print more detailed training status 
        if DEBUG:
            print("**************************** DEBUG INFO START **********************************")
            print(f"Episode {episode} is complete with reward {epsiode_total_reward}")
            print(f"Replay_buffer_length {len(agent.replay_buffer)}")
            print(f"Epsilon for epsilon-greedy is {agent.epsilon}")
            print("**************************** DEBUG INFO END **********************************")
            
        # Analysis prints during training
        if episode % analysis_config.get("ep_freq", 1) == 0:
            run_analysis_tick(agent, env, analysis_config, run_logger, episode)
            if run_logger is not None:
                run_logger.log_rewards(episode_rewards_training,
                                       steps=episode_steps_training)
                run_logger.checkpoint(agent, "latest")
            # The Hankel analysis above rolls out `env` to termination, leaving it in a stale/terminated
            # state. Reset before the next training episode so we don't resume from a hijacked env.
            state, _ = env.reset()
            
        # Check for early stopping 
        if episode > early_stopping_patience_eps:
            ep_avg = sum(episode_rewards_training[-early_stopping_patience_eps:]) / float(early_stopping_patience_eps) 
            if ep_avg > solved_reward:
                print(f"Average trajectory (episode total) reward for {early_stopping_patience_eps} episodes is {ep_avg}")
                print("Triggering Early Stopping !!")
                break

    if run_logger is not None:
        run_logger.log_rewards(episode_rewards_training,
                               steps=episode_steps_training)
        run_logger.checkpoint(agent, "final")
        print(f"run artifacts saved under {run_logger.dir}")
    return episode_rewards_training


def _greedy_episode_return(agent, env, seed: int) -> float:
    state, _ = env.reset(seed=seed)
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        state, reward, terminated, truncated, info = env.step(agent.pi(state))
        total += info.get("raw_reward", reward)
    return total


def policy_iteration_loop(agent, env, no_iterations: int, solved_reward: int,
                          eval_episodes_per_iter: int = 5, np_seed: int = 52,
                          analysis_config: dict | None = None, run_logger=None, DEBUG=False):
    """Classical tabular policy iteration: exhaustive generative-model MC
    evaluation of the current policy, then greedy improvement.

    rewards.csv semantics differ from the DQN loop: one row per PI *iteration*,
    reward = mean greedy-episode return over eval_episodes_per_iter fixed seeds.
    Stops on policy stability (no action changed), solved_reward, or the cap.
    """
    analysis_config = analysis_config or {}
    iteration_rewards = []
    best_avg = float("-inf")
    for iteration in tqdm(range(no_iterations)):
        agent.evaluate_policy(env)
        # Persist this iteration's generative MC rollouts when the agent recorded
        # them (agent.record_rollouts) — for offline truncated/low-rank study.
        if run_logger is not None and getattr(agent, "mc_rollouts", None) is not None:
            run_logger.save_mc_rollouts(iteration, agent.mc_rollouts)
        n_changed = agent.improve_policy()

        # Fixed seeds so the reward curve is comparable across iterations.
        window = [_greedy_episode_return(agent, env, seed=np_seed + e)
                  for e in range(eval_episodes_per_iter)]
        avg = sum(window) / len(window)
        iteration_rewards.append(avg)
        print(f"iteration {iteration}: avg_greedy_reward {avg:.1f}, policy actions changed {n_changed}")
        if DEBUG:
            print(f"Q-table min {agent.Q.min():.3f} max {agent.Q.max():.3f} mean {agent.Q.mean():.3f}")

        if run_logger is not None and avg > best_avg:
            best_avg = avg
            run_logger.checkpoint(agent, "best")

        if iteration % analysis_config.get("ep_freq", 1) == 0:
            run_analysis_tick(agent, env, analysis_config, run_logger, episode=iteration)
            if run_logger is not None:
                run_logger.log_rewards(iteration_rewards)
                run_logger.checkpoint(agent, "latest")
            env.reset()

        if n_changed == 0:
            print("Policy stable — policy iteration converged.")
            break
        if avg >= solved_reward:
            print(f"Average greedy reward {avg} >= {solved_reward}")
            print("Triggering Early Stopping !!")
            break

    if run_logger is not None:
        run_logger.log_rewards(iteration_rewards)
        run_logger.checkpoint(agent, "final")
        print(f"run artifacts saved under {run_logger.dir}")
    return iteration_rewards


