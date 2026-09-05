from agents.q_agent import QAgent
from analysis.low_rank import rank
from analysis.visualisations.heatmaps import plot_matrix_heatmap
import gymnasium as gym
import numpy as np
import torch
import scipy

def _hankel_from_sequence(tau):
    """Square-ish Hankel matrix from a scalar sequence. Rank r of a Hankel matrix
    means the sequence obeys a linear recurrence of order r (a sum of r exponentials),
    so low rank here says the signal along the rollout is a low-order LTI response.
    """
    mid_index = int(len(tau) / 2)  # We can exploit this further and see which hankel matrix is the
    # most low rank and appropriately speed up value iteration.
    return scipy.linalg.hankel(tau[:mid_index + 1], tau[mid_index:])

HANKEL_SEQUENCE_NAMES = ("Hankel Q", "Hankel V", "Hankel A")

def collect_hankel_sequences(agent: QAgent, env: gym.Env, seed: int = 52):
    """Roll out the current policy once and collect the per-step scalar sequences
    the Hankel analysis is built from.

    If the policy network exposes `value_advantage(x) -> (V, A)` (e.g. a dueling
    head), V and A are read straight from its streams. Otherwise V falls back to
    max_a Q and A = Q(s, a_t) - V(s), which is identically zero on greedy steps —
    so for a meaningful advantage Hankel use a dueling network.

    Args:
        agent (QAgent): Deep Q Learning agent.

    Returns:
        dict[str, np.ndarray]: {"Hankel Q": tau_q, "Hankel V": tau_v,
        "Hankel A": tau_a}, each a float array of length H (the rollout length).
    """
    state, _ = env.reset(seed=seed)
    terminated = truncated = False
    tau_q, tau_v, tau_a = [], [], []
    has_dueling_streams = hasattr(agent.policy_net, "value_advantage")
    while not (terminated or truncated):
        # getting the action from current policy
        action = agent.pi(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            q_row = agent.policy_net(state_t)[0]
            q_val = q_row[action].item()
            if has_dueling_streams:
                v, adv = agent.policy_net.value_advantage(state_t)
                v_val = v[0, 0].item()
                a_val = adv[0, action].item()
            else:
                v_val = q_row.max().item()
                a_val = q_val - v_val
        # add to the trajectories
        tau_q.append(q_val)
        tau_v.append(v_val)
        tau_a.append(a_val)
        state, _, terminated, truncated, _ = env.step(action) # get next state

    return {
        "Hankel Q": np.asarray(tau_q, dtype=float),
        "Hankel V": np.asarray(tau_v, dtype=float),
        "Hankel A": np.asarray(tau_a, dtype=float),
    }

def _prefix_lengths(H: int, min_len: int, stride: int):
    """Sub-trajectory prefix lengths: min_len, min_len+stride, ... up to and
    including H (H is always the final length even if the stride would skip it).
    Returns [] when the rollout is shorter than min_len.
    """
    if H < min_len:
        return []
    lengths = list(range(min_len, H + 1, stride))
    if lengths[-1] != H:
        lengths.append(H)
    return lengths

def _milestone_lengths(lengths, n_figures):
    """Pick `n_figures` prefix lengths from `lengths`, equally spaced from the
    first (shortest) to the last (=H) inclusive — the sub-trajectories whose
    spectra we render as figures. Fewer if `lengths` is short / n_figures large.
    """
    if not lengths or n_figures <= 0:
        return set()
    if n_figures >= len(lengths):
        return set(lengths)
    idx = np.linspace(0, len(lengths) - 1, n_figures).round().astype(int)
    return {lengths[i] for i in idx}

def hankel_sweep_analysis(agent: QAgent, env: gym.Env, cfg: dict,
                          run_logger=None, episode=None):
    """Hankel low-rank analysis over several rollouts and sub-trajectories.

    For each of `cfg["n_rollouts"]` seeded rollouts (seed = base_seed + r):
      * optionally dump the raw scalar sequences to <run>/trajectories/;
      * for each selected function build the full-rollout Hankel and log its
        metrics (sub_len = H), rendering the spectrum figure only for rollout 0;
      * if `sub_trajectory.enabled`, sweep growing prefixes 0->L and log a
        metrics-vs-L curve, rendering spectra only at `n_figures` prefix lengths
        equally spaced from the shortest to H (rollout 0 only).

    Self-logging: writes straight to `run_logger` and returns None. With no
    run_logger it prints a compact summary and shows the rendered figures inline.

    Config keys (all optional, with defaults):
        n_rollouts (int=5), base_seed (int=52),
        functions (list[str]=all three sequence names),
        save_trajectories (bool=False),
        save_heatmaps (bool=False): also save a heatmap of each rendered Hankel
            matrix (at the same milestone lengths as the spectra, rollout 0),
        sub_trajectory: {enabled (bool=False), min_len (int=8), stride (int=16),
                         n_figures (int=5)}
    """
    n_rollouts = cfg.get("n_rollouts")
    base_seed = cfg.get("base_seed", 52)
    Hankel_objects = cfg.get("functions", list(HANKEL_SEQUENCE_NAMES))
    save_trajectories = cfg.get("save_trajectories", True)
    # Opt-in: alongside each rendered spectrum, also save a heatmap of that Hankel
    # matrix (default off so existing experiments' artifacts are unchanged).
    save_heatmaps = cfg.get("save_heatmaps", False)
    sub = cfg.get("sub_trajectory", {}) or {}
    sub_enabled = sub.get("enabled", True)
    min_len = sub.get("min_len")
    stride = sub.get("stride")
    n_figures = sub.get("n_figures")

    for r in range(n_rollouts):
        seed = base_seed + r # We also change the seed to obtain some variability ( Not sure if this is the best idea)
        seqs = collect_hankel_sequences(agent, env, seed)
        if save_trajectories and run_logger is not None:
            run_logger.save_trajectory(episode, seed, seqs)

        for name in Hankel_objects:
            tau = seqs[name]
            H = len(tau)
            lengths = _prefix_lengths(H, min_len, stride) if sub_enabled else [H]
            # Spectra are rendered only for rollout 0, at n_figures equally-spaced
            fig_lengths = _milestone_lengths(lengths, n_figures) if r == 0 else set()

            for L in lengths:
                hk = _hankel_from_sequence(tau[:L])
                # one SVD per matrix: the spectrum feeds both the rendered
                # figure and the sv_* columns of hankel_sweep.csv (the
                # sigma_i / sigma_{i-1} decay analysis reads those)
                s_vals, metrics = rank.spectrum_and_metrics(hk)
                if L in fig_lengths:
                    label = f"{name} full r{r}" if L == H else f"{name} sub {L}"
                    save_to = (run_logger.figure_path(_figure_name(name, L, H, r), episode)
                               if run_logger is not None else None)
                    rank.plot_matrix_spectra(s_vals, label, save_to=save_to,
                                             show=run_logger is None)
                    if save_heatmaps:
                        hm_save_to = (run_logger.figure_path(
                            _figure_name(name, L, H, r) + " heatmap", episode)
                            if run_logger is not None else None)
                        plot_matrix_heatmap(hk, label, save_to=hm_save_to,
                                            show=run_logger is None)

                if run_logger is not None:
                    run_logger.log_hankel_sweep(episode, name, r, seed, L,
                                                *metrics, s_vals=s_vals)
                else:
                    _print_sweep_row(name, r, seed, L, metrics)

def _figure_name(name, L, H, rollout):
    """Figure filename stem for a rendered spectrum"""
    return f"{name} rollout{rollout}" if L == H else f"{name} sub{L:06d}"

def _print_sweep_row(name, rollout, seed, sub_len, metrics):
    """One-line summary used when no run_logger is attached (inline mode)."""
    eff_rank, stable_rank, spikiness = metrics[0], metrics[1], metrics[2]
    print(f"[hankel_sweep] {name} rollout={rollout} seed={seed} sub_len={sub_len} "
          f"eff_rank={eff_rank} stable_rank={stable_rank:.2f} spikiness={spikiness:.2f}")
