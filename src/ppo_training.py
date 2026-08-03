"""On-policy rollout collection + training loop for PPOAgent."""
import numpy as np
from tqdm import tqdm


def _collect_rollout(agent, env, state, ep_ret, window_len):
    n = agent.rollout_steps
    obs_dim = env.observation_space.shape[0]
    buf = {"obs": np.zeros((n, obs_dim), np.float32), "acts": np.zeros(n, np.int64),
           "logps": np.zeros(n, np.float32), "rews": np.zeros(n, np.float64),
           "values": np.zeros(n, np.float64)}
    seg_bounds, seg_terminal, seg_boot = [], [], []
    finished_returns = []
    seg_start = 0
    for t in range(n):
        a, logp, v = agent.act(state)
        buf["obs"][t] = state
        buf["acts"][t], buf["logps"][t], buf["values"][t] = a, logp, v
        state, r, terminated, truncated, _ = env.step(a)
        buf["rews"][t] = r
        ep_ret += r
        if terminated or truncated:
            seg_bounds.append((seg_start, t + 1))
            seg_terminal.append(bool(terminated))
            # env-truncated (TimeLimit): bootstrap with V of the final state
            seg_boot.append(0.0 if terminated else
                            agent.act_and_value_only(state))
            finished_returns.append(ep_ret)
            agent.record_episode_return(ep_ret)
            ep_ret = 0.0
            state, _ = env.reset()
            seg_start = t + 1
    if seg_start < n:  # rollout-cut segment: bootstrap with V(next state)
        seg_bounds.append((seg_start, n))
        seg_terminal.append(False)
        seg_boot.append(agent.act_and_value_only(state))
    # within-episode contiguous windows for the critic penalty
    wins = []
    for (a0, b0), term in zip(seg_bounds, seg_terminal):
        hi = b0 - (1 if term else 0)  # drop the pre-terminal anchor ambiguity
        for s in range(a0, hi - window_len + 1):
            wins.append(np.arange(s, s + window_len))
    buf["seg_bounds"] = seg_bounds
    buf["seg_terminal"] = seg_terminal
    buf["seg_boot_value"] = seg_boot
    buf["ep_windows"] = np.array(wins, dtype=np.int64) if wins else np.zeros((0, window_len), np.int64)
    return buf, state, ep_ret, finished_returns


def ppo_training_loop(agent, env, no_episodes, solved_reward,
                      early_stopping_patience_eps=50, np_seed=52,
                      no_eps_to_avg=10, DEBUG=False):
    """Runs PPO updates until no_episodes episodes have finished (or solved
    early-stop). Returns per-episode returns, matching dqn_training_loop."""
    state, _ = env.reset(seed=np_seed)
    ep_ret = 0.0
    episode_rewards = []
    pbar = tqdm(total=no_episodes)
    solved_streak = 0
    while len(episode_rewards) < no_episodes:
        buf, state, ep_ret, finished = _collect_rollout(
            agent, env, state, ep_ret, agent.window_len)
        agent.update(buf)
        episode_rewards.extend(finished)
        pbar.update(len(finished))
        if len(episode_rewards) >= no_eps_to_avg:
            avg = np.mean(episode_rewards[-no_eps_to_avg:])
            if DEBUG:
                print(f"eps {len(episode_rewards)} avg {avg:.1f} diag {agent.diag}")
            if avg >= solved_reward:
                solved_streak += len(finished)
                if solved_streak >= early_stopping_patience_eps:
                    break
            else:
                solved_streak = 0
    pbar.close()
    return episode_rewards[:no_episodes]


def greedy_episode_return(agent, env, seed):
    state, _ = env.reset(seed=seed)
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        state, r, terminated, truncated, info = env.step(agent.act_greedy(state))
        total += info.get("raw_reward", r)
    return total
