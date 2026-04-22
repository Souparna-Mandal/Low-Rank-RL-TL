# `low_rank_rl.envs`

Thin wrapper layer over Gymnasium that gives every agent / analysis tool
three things it can rely on:

- a **`Discrete` action space** (for DQN, Q-learning, etc.);
- **bounded, with optionally unit-scale observations** (clean NN inputs,
  well-defined bin edges)
- an **optional state discretiser** that puts observations to a fixed
  grid of bin centres — used by the analysis layer to enumerate the full
  tabular state space.

## `make_env(env_id, …)`

```python
from low_rank_rl.envs import make_env

env = make_env("Acrobot-v1")                                       # 6-D discretised [7]*6
env = make_env("MountainCarContinuous-v0", n_discrete_actions=21)  # 2-D discretised 40x40
env = make_env("Pendulum-v1", n_state_bins=[11, 11, 15])           # override the default grid
env = make_env("GridWorld-v0", size=(5, 5))                        # tabular, no wrappers
```

Defaults per env live in `_ENV_DEFAULTS`:

| Env                       | normalise | action space size  | state grid (bins per dim)     |
|---------------------------|-----------|------------|-------------------------------|
| `Acrobot-v1`              | yes       |  3   | `[7, 7, 7, 7, 7, 7]`          |
| `CartPole-v1`             | no        |  2   | `[10, 10, 10, 10]`            |
| `Pendulum-v1`             | yes       | 15 bins    | `[11, 11, 15]`                |
| `MountainCarContinuous-v0`| yes       | 21 bins    | `[40, 40]`                    |
| `LunarLander-v2`          | no        |  4   | — (continuous)                |
| `HalfCheetah / Hopper / Ant-v4` | yes | 11 bins    | — (continuous)                |
| `GridWorld-v0`            | no        |  4   | — (already tabular)           |

Any default can be overridden by passing the same kwarg to `make_env`. Unknown env ids fall through with safe defaults (no wrappers, so no discretisation or response clipping).

The factory sets `env.metadata` for each instance (copied from the
class-level dict to avoid cross-instance leakage):

- `action_type` = `"discrete_original"` or `"discrete_wrapped"`
- `obs_type`    = `"discretized"` or `"continuous"`

## `DiscreteActionWrapper`

Maps a 1-D continuous action Box onto `n_actions` evenly-spaced points:

$$
a_i = a_\text{low} + i \cdot \frac{a_\text{high} - a_\text{low}}{n_\text{actions} - 1}, \quad i \in \{0, \dots, n_\text{actions} - 1\}.
$$

Asserts the underlying space is `Box` with `shape == (1,)`. Exposes
`action_values` (1-D float array) so tensor code can recover the control
value for each discrete action id.

## `NormalizeObsWrapper`

Rescales observations to $[-1, 1]$ using the env's declared bounds:

$$
\tilde s_d = 2 \cdot \frac{\operatorname{clip}(s_d; \, l_d, h_d) - l_d}{h_d - l_d} - 1.
$$

- **Infinite bounds** (e.g. Pendulum's angular velocity) are clipped to
  ±`_CLIP_FALLBACK = 5.0` before rescaling.
- `_range` is floor-clipped at `1e-8` to avoid divide-by-zero.

## `DiscretizeObsWrapper`

Snaps each continuous observation dim to the centre of one of `n_bins[d]`
uniform bins. This is **active during training and analysis** — the agent
still receives a float observation, this makes the
set of possible observations finite, so we can enumerate the full
tabular state space.

Exposes:

- `bin_edges` / `bin_centers` — lists of per-dim 1-D arrays.
- `n_bins` — per-dim list of ints.
- `obs_to_index(obs) -> tuple[int, ...]` — O(1) bin lookup used by the
  successor-matrix code.

Note: this is a pure discretisation of the raw observation — trigonometric
encodings like `(cos θ, sin θ)` are binned in their raw form, which could lead to bins that maybe always empty.

## `GridWorldEnv`
Thhis is just a standard GridWorld implementation. It's a
Tabular $n \times m$ gridworld with configurable walls, slip probability,
max step count, and (start, goal) coordinates. `state_grid()` returns an
`(N, 2)` array of every reachable cell.

Step reward is $-1$ until the goal (which gives $0$ and terminates).

## `find_obs_discretizer(env)`

Walks the wrapper chain via `getattr(cur, "env", None)` and returns the
first `DiscretizeObsWrapper` instance it finds, or `None`. Used by the
analysis layer to decide between canonical-grid enumeration and
Monte-Carlo sampling to built Q table and Value functions.

## Tests

`tests/test_envs.py` covers the registry, each wrapper in isolation,
`GridWorldEnv` dynamics, and `find_obs_discretizer` going through the wrapper
stack. Note the Tests have to be taken with a grain of salt as these were heavily AI generated without much review for sanity checks. 
