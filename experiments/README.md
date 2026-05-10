# `experiments/`

Thin wrapper that wires a YAML config → `make_env` → `build_agent` → training
loop → full analysis. Produces a folder of PNGs plus a saved agent checkpoint.

## Running

### From the command line

```
python experiments/run.py --config experiments/configs/acrobot_dqn.yaml
python experiments/run.py --config experiments/configs/acrobot_ppo.yaml --episodes 200
```

The optional `--episodes` flag overrides `training.n_episodes` in the config.

### Interactively, from a notebook

`experiments/notebook.ipynb` mirrors `run.py` cell-by-cell: load a YAML (or
build `cfg` in Python), build the env + agent, train with a **tqdm progress
bar and a live reward-vs-episode plot**, then step through every analysis —
Q-matrix spectrum, HOSVD, Hankel, successor shift, value heatmap — with plots
rendered inline.

The CLI and notebook share the same training loop in
`low_rank_rl/training.py`. The DQN loop follows the
[PyTorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
and the PPO loop follows
[OpenAI Spinning Up's PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

```
jupyter notebook experiments/notebook.ipynb
```

The notebook auto-detects the repo root from any working directory, so it
works wherever you launch Jupyter. A quick 50-episode Acrobot-DQN run takes
~1–2 minutes on a laptop CPU.

## Config schema

```yaml
env_id: "Acrobot-v1"            # Gymnasium env id registered in low_rank_rl.envs
env_kwargs:                     # forwarded to make_env; overrides per-env defaults
  n_discrete_actions: null      # wrap a Box action space to Discrete(n) (null = no wrap)
  normalize_obs:      null      # null = use env default
  discretize_obs:     null      # null = use env default
  n_state_bins:       null      # per-dim bin counts; null = env default

agent: "dqn"                    # one of: dqn, ppo, qlearning, sarsa, monte_carlo
agent_kwargs:                   # forwarded to the agent constructor
  hidden: 128
  gamma:  0.99

training:
  n_episodes: 500               # outer loop length
  rank_checkpoint_every: 50     # compute Q-matrix rank every N episodes
  n_rank_samples: 500           # state probe count; IGNORED when the env is
                                # discretised (the full canonical grid is used)

analysis:                       # everything below is optional
  value_tensor_bins: null       # null = use the wrapper's canonical grid;
                                # set an int to force the sample-and-average path
  value_tensor_dims: [0, 1, 2, 3]   # which obs dims to use as tensor axes
  hankel_steps: 500             # max trajectory length for Hankel analysis
  hankel_n_rows: null           # defaults to T // 2

output:
  save_dir: "out/acrobot_dqn"
  save_agent: true              # persist the trained agent via agent.save()
```

## Training loops

Three training drivers are exposed to match each agent's update model:

- **`train_step_based`** — DQN / Q-learning / SARSA. Calls
  `agent.update(s, a, r, s′, done)` on every step. Passes `next_state=None`
  on terminal transitions (Gymnasium convention used by `DQNAgent`).
- **`train_episode_based`** — Monte Carlo. Calls `agent.update` to record
  each step and `agent.end_episode()` to trigger the first-visit MC update.
- **`train_ppo`** — Collects a rollout via `agent.update` and calls
  `agent.end_episode(last_value)` to run GAE + PPO updates. When the episode
  was *truncated* (not `terminated`), `last_value` is set to the critic's
  estimate of the final state; on true termination it's 0.

All three return `(durations, rank_history)` where `rank_history` is a list
of dicts sampled every `rank_checkpoint_every` episodes.

## What `run_analysis` produces

Given a trained agent, the function writes the following to `output.save_dir`:

| File | Source |
|---|---|
| `training_durations.png`  | `plot_episode_durations` |
| `rank_vs_episode.png`     | `plot_rank_vs_episode` (if checkpoints were taken) |
| `q_spectrum.png`          | `plot_singular_value_spectrum` on `compute_rank_metrics` |
| `hosvd_spectra.png`       | `plot_hosvd_spectra` on `hosvd_spectra(build_value_tensor(…))` |
| `hankel_value.png`        | `plot_hankel_spectrum` for the value-trajectory Hankel |
| `hankel_q_taken.png`      | ditto, for the Q-taken trajectory |
| `hankel_policy.png`       | ditto, for the greedy-action trajectory |
| `hankel_actions.png`      | ditto, for the realised-action trajectory |
| `successor_shift.png`     | `plot_shift_comparison` for $M$ vs $\tilde M$ |
| `value_heatmap.png`       | `plot_value_heatmap` on dims (0, 1) |
| `agent.pt`                | `agent.save(…)` (if `save_agent: true`) |

On discretised envs the Q-rank and HOSVD steps see the **full canonical
grid** (every bin centre), so the rank metrics are exact — there's no
sampling noise, and `n_rank_samples` is ignored for those two analyses.
The successor-measure step is the only $O(N^2 T)$ cost, so it takes a
random subset of the canonical grid via `canonical_subsample(env, N)`
with $N = \min(64, \text{n\_rank\_samples})$.

## Configs shipped in `configs/`

- `acrobot_dqn.yaml`     — DQN on Acrobot-v1 (6-D obs discretised `[7]*6`)
- `acrobot_ppo.yaml`     — PPO on Acrobot-v1
- `mountaincar_dqn.yaml` — DQN on MountainCarContinuous-v0 (2-D obs discretised `40×40`)

Edit these to start a new experiment; prefer copying an existing file over
constructing one from scratch.
