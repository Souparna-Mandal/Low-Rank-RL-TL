# `low_rank_rl.agents`

All agents implement a single contract, `BaseAgent`, which exposes three
abstract methods plus a few convenience helpers. This allows every analysis
tool in the package to work with any agent unchanged.

## `BaseAgent` contract

| Method | Signature | Purpose |
|---|---|---|
| `act`        | `(state, training=True) -> int`      | Select an action. `training=True` explores (e.g. ε-greedy / sampling); `training=False` is deterministic / greedy. |
| `update`     | `(s, a, r, s′, done, …) -> dict`     | Consume one transition. Returns loggable scalars (`{}` if no update happened, e.g. replay buffer still filling). |
| `q_matrix`   | `(states[N, obs_dim]) -> (N, nA)`    | Return Q(s, a) over a batch **without mutating any internal state**. This is what every analysis routine calls. |
| `value_vector`  | `(states) -> (N,)`                | `max_a Q(s, a)`. |
| `policy_vector` | `(states) -> (N,) int`            | `argmax_a Q(s, a)`. |
| `save` / `load` | `(path)`                          | Persistence (optional; raises NotImplementedError otherwise). |

## Algorithms

### Tabular algorithms (`q_learning.py`, `sarsa.py`, `monte_carlo.py`)

All three discretise continuous observations by binning each dimension with
`np.linspace(low, high, n_bins)` and indexing with `np.digitize`. Infinite
observation bounds are clipped to ±1e4 before binning.

**Q-learning — off-policy TD(0):**

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
$$

**SARSA — on-policy TD(0):**

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma Q(s', a') - Q(s, a) \Big]
$$

where $a'$ is the action actually taken by the current ε-greedy policy at
$s'$. If `next_action` is not supplied it is sampled from the current policy.

**Monte Carlo — first-visit on-policy:**

$$
Q(s, a) \leftarrow Q(s, a) + \frac{1}{n(s,a)} \big( G_t - Q(s, a) \big)
$$

where $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$ is computed at the end of the
episode. Only the first visit to each $(s, a)$ contributes. The incremental
mean avoids storing all returns. Call order per episode:
`act` / `env.step` / `update` (records the step) … then `end_episode()`
triggers the table update.

### Deep Q-Network (`dqn.py`)

A 3-layer MLP Q-network with:

- **Experience replay** of capacity `buffer_capacity`.
- **Soft Polyak target update**: $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$.
- **ε-greedy schedule** with exponential decay:

$$
\epsilon(t) = \epsilon_\mathrm{end} + (\epsilon_\mathrm{start} - \epsilon_\mathrm{end}) \, e^{-t/\mathrm{decay}}
$$

- **Huber loss** (`smooth_l1_loss`) on the TD error.
- **Gradient clipping** at ±100 element-wise.

A mini-batch of transitions yields

$$
y_i =
\begin{cases}
r_i & \text{terminal step} \\
r_i + \gamma \max_{a'} Q_{\theta'}(s'_i, a') & \text{otherwise}
\end{cases}
$$

and the loss is $\mathrm{Huber}\!\big(Q_\theta(s_i, a_i) - y_i\big)$.
`QNetwork` is intentionally split from `DQNAgent` so that a low-rank-factored
version can be dropped in later.

### PPO (`ppo.py`)

Separate actor and critic, each a 2-layer MLP with Tanh activations. Per
rollout we compute **Generalised Advantage Estimation**:

$$
\delta_t = r_t + \gamma V(s_{t+1}) \, m_t - V(s_t), \qquad
\hat A_t = \sum_{k \ge 0} (\gamma \lambda)^k \delta_{t+k} \, m_{t+k}
$$

where $m_t = 0$ at terminal steps and $1$ otherwise, and
$\hat R_t = \hat A_t + V(s_t)$.

Over `ppo_epochs` passes of `mini_batch_size` mini-batches we minimise

$$
\mathcal L = - \mathbb{E} \Big[ \min\!\big(\rho_t \hat A_t, \operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \hat A_t \big) \Big] + c_V \|V_\theta - \hat R\|^2 - c_H \, \mathcal H[\pi_\theta]
$$

with $\rho_t = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_\text{old}}(a_t|s_t))$.

**`q_matrix` approximation for rank analysis.** PPO has no explicit Q. We
define a rank-analysis proxy

$$
Q(s, a) \approx V(s) + \underbrace{\big( \ell(a \mid s) - \bar\ell(\cdot \mid s) \big)}_{\text{centred logits}}
$$

which makes the row mean equal to $V(s)$. This is used *only* by the rank /
spectral tools and never feeds back into training.

The caller flow is:

1. `agent.act(s)` selects an action and caches log-prob + value internally.
2. `agent.update(…)` pushes the step into the rollout buffer.
3. After the episode, `agent.end_episode(last_value)` runs GAE + PPO updates and clears the buffer. `last_value` should be $V(s_T)$ if the episode was *truncated* (and 0 on a true terminal transition).

## Testing

See `tests/test_agents.py`. Each agent has unit tests covering:

- `act` returns a valid action id in `[0, n_actions)`.
- `act(training=False)` is deterministic given a state.
- `q_matrix` returns the correct shape, dtype `float64`, and does not mutate
  ε / step counters.
- `update` returns `{}` when no update occurred (e.g. replay buffer still
  filling, Monte Carlo mid-episode).
- `save` / `load` round-trips preserve Q values exactly.
- Agent-specific invariants:
  - DQN: ε decreases with steps.
  - SARSA: on-policy target differs from Q-learning's max-target.
  - Monte Carlo: first-visit rule; buffer cleared at episode end.
  - PPO: `q_matrix` rows are centred around $V(s)$.
