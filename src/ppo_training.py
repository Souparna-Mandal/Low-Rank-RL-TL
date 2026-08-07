"""On-policy rollout collection + training loop for PPOAgent."""
import contextlib

import numpy as np
from tqdm import tqdm


def raw_return(info, fallback):
    """The episode's UNNORMALISED return.

    RecordEpisodeStatistics is installed inside the running normalisers (see
    base_env), so info["episode"]["r"] carries the true reward sum while
    env.step() hands the agent the normalised one. With normalisation off the
    wrapper is absent and the accumulated fallback already is the raw return.
    """
    ep = info.get("episode") if info else None
    if ep is None or ep.get("r") is None:
        return float(fallback)
    return float(np.asarray(ep["r"]).item())


@contextlib.contextmanager
def frozen_running_stats(env):
    """Stop NormalizeObservation/NormalizeReward updating their statistics.

    Evaluation must not move them: a greedy rollout would otherwise shift the
    mean/std that the training policy's observations are standardised against,
    so measuring the agent would change the agent.
    """
    frozen, e = [], env
    while e is not None:
        if hasattr(e, "update_running_mean"):
            frozen.append((e, e.update_running_mean))
            e.update_running_mean = False
        e = getattr(e, "env", None)
    try:
        yield
    finally:
        for wrapper, prev in frozen:
            wrapper.update_running_mean = prev


def make_rollout_buffer(agent, obs_dim, n):
    """Empty rollout buffer of n steps.

    The action column is (n,) int64 for a Discrete agent and (n, act_dim)
    float32 for a continuous one. Variants that ship their own collect_rollout
    should build their buffer here so that dtype rule stays in one place.
    """
    acts = (np.zeros((n, agent.act_dim), np.float32) if agent.continuous
            else np.zeros(n, np.int64))
    return {"obs": np.zeros((n, obs_dim), np.float32), "acts": acts,
            "logps": np.zeros(n, np.float32), "rews": np.zeros(n, np.float64),
            "values": np.zeros(n, np.float64)}


def _collect_rollout(agent, env, state, ep_ret):
    n = agent.rollout_steps
    obs_dim = env.observation_space.shape[0]
    buf = make_rollout_buffer(agent, obs_dim, n)
    seg_bounds, seg_terminal, seg_boot = [], [], []
    finished_returns, finished_raw = [], []
    seg_start = 0
    for t in range(n):
        a, logp, v = agent.act(state)
        buf["obs"][t] = state
        buf["acts"][t], buf["logps"][t], buf["values"][t] = a, logp, v
        state, r, terminated, truncated, info = env.step(a)
        buf["rews"][t] = r
        ep_ret += r
        if terminated or truncated:
            seg_bounds.append((seg_start, t + 1))
            seg_terminal.append(bool(terminated))
            # env-truncated (TimeLimit): bootstrap with V of the final state
            seg_boot.append(0.0 if terminated else
                            agent.act_and_value_only(state))
            finished_returns.append(ep_ret)
            finished_raw.append(raw_return(info, ep_ret))
            ep_ret = 0.0
            state, _ = env.reset()
            seg_start = t + 1
    if seg_start < n:  # rollout-cut segment: bootstrap with V(next state)
        seg_bounds.append((seg_start, n))
        seg_terminal.append(False)
        seg_boot.append(agent.act_and_value_only(state))
    buf["seg_bounds"] = seg_bounds
    buf["seg_terminal"] = seg_terminal
    buf["seg_boot_value"] = seg_boot
    # Raw (unnormalised) counterpart of finished_returns, same length and
    # order. Rides on buf so the variants' collect_rollout signature is
    # unchanged; equals finished_returns when normalisation is off.
    buf["raw_returns"] = finished_raw
    return buf, state, ep_ret, finished_returns


def ppo_training_loop(agent, env, no_episodes, solved_reward,
                      early_stopping_patience_eps=50, np_seed=52,
                      no_eps_to_avg=10, DEBUG=False, progress=True,
                      return_raw=False):
    """Runs PPO updates until no_episodes episodes have finished (or solved
    early-stop). Returns per-episode returns, matching dqn_training_loop.
    Pass progress=False to silence the tqdm bar (tests, batch scripts).

    The returned returns are what the agent optimises, i.e. normalised when
    normalise.running.reward is on. return_raw=True instead returns
    (returns, raw_returns) so both streams can be reported; the two are equal
    when reward normalisation is off. solved_reward is compared against the
    RAW returns, so a solved threshold keeps its physical meaning either way.
    """
    state, _ = env.reset(seed=np_seed)
    ep_ret = 0.0
    episode_rewards, raw_rewards = [], []
    pbar = tqdm(total=no_episodes, disable=not progress)
    solved_streak = 0
    empty_rollouts = 0
    while len(episode_rewards) < no_episodes:
        buf, state, ep_ret, finished = _collect_rollout(agent, env, state, ep_ret)
        agent.update(buf)
        episode_rewards.extend(finished)
        raw_rewards.extend(buf.get("raw_returns") or finished)
        pbar.update(len(finished))
        # Safety net for envs without an episode cap: a long streak of
        # rollouts with no finished episode means the env never terminates.
        if finished:
            empty_rollouts = 0
        else:
            empty_rollouts += 1
            if empty_rollouts >= 100:
                pbar.close()
                raise RuntimeError(
                    f"no episode finished in {empty_rollouts} consecutive "
                    f"rollouts ({empty_rollouts * agent.rollout_steps} steps); "
                    "the environment appears to never terminate — cap it with "
                    "the time_limit environment config key")
            continue
        if len(episode_rewards) >= no_eps_to_avg:
            # Judge "solved" on the raw returns: a normalised average has no
            # fixed scale, so solved_reward would otherwise be meaningless.
            avg = np.mean(raw_rewards[-no_eps_to_avg:])
            if DEBUG:
                print(f"eps {len(episode_rewards)} avg {avg:.1f}")
            if avg >= solved_reward:
                solved_streak += len(finished)
                if solved_streak >= early_stopping_patience_eps:
                    break
            else:
                solved_streak = 0
    pbar.close()
    if return_raw:
        return episode_rewards[:no_episodes], raw_rewards[:no_episodes]
    return episode_rewards[:no_episodes]


def greedy_episode_return(agent, env, seed, max_steps=100_000, raw=True):
    """One argmax-policy episode's return. max_steps caps evaluation on envs
    without an episode limit (see the time_limit config key).

    Running normalisation statistics are frozen for the episode, so evaluating
    cannot perturb the agent. raw=True reports the unnormalised return, which
    is the one comparable across runs and against a solved threshold.
    """
    with frozen_running_stats(env):
        state, _ = env.reset(seed=seed)
        total, terminated, truncated = 0.0, False, False
        steps = 0
        info = None
        while not (terminated or truncated):
            state, r, terminated, truncated, info = env.step(
                agent.act_greedy(state))
            total += r
            steps += 1
            if steps >= max_steps:
                break
        return raw_return(info, total) if raw else total
