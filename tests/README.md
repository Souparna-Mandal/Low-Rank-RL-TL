# `tests/`

Pytest suite covering every public module of `low_rank_rl`. Run from the
project root:

```
pytest -q
pytest tests/test_agents.py -v      # one module
pytest tests/test_agents.py::TestDQNAgent::test_q_matrix_shape  # one test
```

Matplotlib tests use the `Agg` backend (`matplotlib.use("Agg")`) so the suite
runs headlessly.

## What each file tests

### `test_agents.py`

Per-agent test classes: `TestDQNAgent`, `TestQLearningAgent`, `TestSarsaAgent`,
`TestMonteCarloAgent`, `TestPPOAgent`. Every agent is checked against the
`BaseAgent` contract:

- `act` returns a valid action id in `[0, n_actions)`.
- `act(training=False)` is deterministic on the same state.
- `q_matrix(states)` returns shape `(N, n_actions)` with dtype `float64`.
- `q_matrix` does **not** mutate training state (ε / step counters are
  unchanged).
- `save` / `load` produce identical `q_matrix` output on a fresh agent.

Agent-specific invariants:

- **DQN**: `update` returns `{}` while the replay buffer fills; ε decreases
  strictly as `act(training=True)` is called.
- **Q-learning**: unseen states return zeros; `decay_epsilon` strictly
  decreases ε.
- **SARSA**: on-policy target uses `next_action`; with controlled Q values
  its update diverges from Q-learning's max-target. Terminal steps ignore
  $Q(s', a')$.
- **Monte Carlo**: `update` returns `{}` mid-episode; `end_episode` returns
  `{"mean_update", "episode_len", "epsilon"}` and clears the buffer. First-
  visit rule is verified by inserting two visits of the same $(s, a)$ in one
  episode and checking the return count stays at 1.
- **PPO**: `q_matrix` rows are centred around $V(s)$; rollout buffer is
  cleared after `end_episode`; `end_episode` returns a loss.

### `test_envs.py`

- `registered_envs()` returns a non-empty list including `"Acrobot-v1"` and
  `"MountainCarContinuous-v0"`.
- `make_env` wraps continuous-action envs to `Discrete` and sets
  `metadata["action_type"]` appropriately.
- `normalize_obs=True` override produces observations in $[-1, 1]$.
- `DiscreteActionWrapper`:
  - action space is `Discrete(n_actions)`;
  - endpoints of `action_values` match the underlying Box bounds;
  - non-Box action spaces are rejected via `AssertionError`.
- `NormalizeObsWrapper`:
  - resets of MountainCarContinuous yield observations in $[-1, 1]$
    within `1e-5` tolerance;
  - Pendulum's infinite bounds do not produce non-finite observations.

### `test_analysis.py`

- **`rank.py`** — stable rank equals 1 for an exact rank-1 matrix
  (constant row), `normalised_numerical_rank == 1` for a random full-rank
  matrix, bounds $1 \le \text{stable}, \text{effective} \le \min(m, n)$,
  spectral gap $\ge 0$, `summary` contains the expected keywords,
  `sample_states` returns `(n, obs_dim)`.
- **`tensor.py`** — `_mode_unfold` shapes on a 3-tensor, `hosvd_spectra`
  keys, descending singular values per mode, positive stable ranks,
  `build_value_tensor` produces the expected shape without NaNs.
- **`hankel.py`** — Hankel antidiagonal identity; a constant sequence gives
  rank 1; a pure sinusoid has no more than 3 numerically significant
  singular values; `collect_trajectory` returns all expected keys of
  consistent length; `hankel_rank_metrics` returns a `HankelMetrics`
  dataclass; DMD modes and eigenvalues have the correct shape and
  `eigenvalues.dtype` is complex.
- **`successor.py`** — shifted matrix preserves shape; its column means are
  zero within `1e-10`; shifting cannot increase numerical rank
  (projection property); a matrix of pure stationary rows shifts to zero;
  a "stationary (rank-1) + rank-2 signal" matrix shifts to numerical rank
  at most 2.

### `test_visualization.py`

Every plotting function is called and the returned object is asserted to be
a `matplotlib.figure.Figure`, with edge cases:

- `plot_episode_durations` on a series shorter than the rolling window.
- `plot_singular_value_spectrum` with and without log-scale.
- `plot_hosvd_spectra` with a single mode and with multiple modes (including
  custom `dim_labels`).
- `plot_q_heatmap` across every discrete action on Acrobot-v1.

All figures are closed with `plt.close(fig)` after assertion to keep the
test process memory bounded.

## Adding a new agent

To keep the new agent compatible with the analysis layer, add a test class
in `test_agents.py` mirroring the existing structure:

```python
class TestMyAgent:
    def setup_method(self):
        self.agent = MyAgent(N_OBS, N_ACTIONS, ...)

    def test_act_valid_action(self):    ...
    def test_q_matrix_shape(self):      ...
    def test_q_matrix_immutable(self):  ...
    def test_save_load_roundtrip(self): ...
```

then, if the analysis code needs anything special from it, add the
corresponding integration check in `test_analysis.py`.
