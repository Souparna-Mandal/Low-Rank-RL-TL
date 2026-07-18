from agents import q_agent
from analysis.low_rank import rank, tabular_q_matrix, hankel_policy
from analysis.registry import resolve_methods
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def dqn_training_loop(agent: q_agent.QAgent, env: gym.Env,
                      no_episodes: int, target_network_update_steps: int,
                      train_frequency_steps: int, use_episode_training: bool, solved_reward: int,
                      warmup_steps: int = 0, early_stopping_patience_eps: int = 50,
                      no_episodes: int, target_network_update_steps: int,
                      train_frequency_steps: int, use_episode_training: bool, solved_reward: int,
                      warmup_steps: int = 0, early_stopping_patience_eps: int = 50,
                      np_seed: int = 52, no_eps_to_avg: int = 10,
                      analysis_config: dict = {},
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
        analysis_config (dict, optional): _description_. Defaults to {}.
        DEBUG (bool, optional): _description_. Defaults to False.
        run_logger (RunLogger, optional): when provided, analysis figures are
            saved to its run directory instead of rendered inline, rank stats go
            to rank_stats.csv, rewards to rewards.csv, and checkpoints are kept
            (latest at every analysis tick, best on new reward-window high, final
            on return). Defaults to None (old behaviour, everything inline).

    Returns:
        _type_: _description_
    """
    state, info = env.reset(seed = np_seed)
    s_tn_upd, s_train, step_count = 0,0,0
    episode_rewards_training = []
    best_window_avg = float("-inf")
    for episode in tqdm(range(no_episodes)):
        # Do a policy rollout and explore with the current agent
        epsiode_total_reward = 0
        terminated = truncated = False
        # This is one rollout from the current policy pi = DQN(theeta) 
        # which is updated once every episode
        while not (terminated or truncated):
            action = agent.pi(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_buffer(state, action, reward, next_state, terminated) if not atari else  agent.update_buffer_atari(state, action, reward, next_state, terminated)
            state = next_state
            epsiode_total_reward += reward
            
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
                agent.train() # train every train_frequency_steps steps

        # reset the environment for next episode
        state, _ = env.reset()
        episode_rewards_training.append(epsiode_total_reward)
        
        # We will do the training only when the episode had ended... and ensure that at least train_frequency_steps have passed since the last training
        # This is mainly relevant only when use_episode_training is True
        if s_train >= train_frequency_steps:
            s_train = 0
            # train the agent 
            agent.train() # train every train_frequency_steps steps
        
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
        if episode % analysis_config["ep_freq"] == 0:
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
            if run_logger is not None:
                run_logger.log_rewards(episode_rewards_training)
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
        run_logger.log_rewards(episode_rewards_training)
        run_logger.checkpoint(agent, "final")
        print(f"run artifacts saved under {run_logger.dir}")
    return episode_rewards_training
