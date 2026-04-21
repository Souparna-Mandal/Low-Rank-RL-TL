# `low_rank_rl.envs`

Thin Gymnasium factory + two wrappers that make every agent speak the same
interface:

- **discrete action space** (because every agent in this package consumes
  discrete actions);
- **bounded, unit-scale observations** when requested, so both tabular bin
  edges and neural-net inputs are well-conditioned.

## `make_env(env_id, …)`

```
from low_rank_rl.envs import make_env

env = make_env("Acrobot-v1")
env = make_env("MountainCarContinuous-v0", n_discrete_actions=21)
env = make_env("Pendulum-v1", n_discrete_actions=15, normalize_obs=True)
```

Defaults per environment live in `_ENV_DEFAULTS`:

| Env | normalise obs | discretise to |
|---|---|---|
| Acrobot-v1, CartPole-v1, LunarLander-v2 | no  | already discrete |
| MountainCarContinuous-v0                | yes | 21 bins |
| Pendulum-v1                             | yes | 15 bins |
| HalfCheetah-v4, Hopper-v4, Ant-v4       | yes | 11 bins |

Overrides can be passed directly to `make_env`. Unknown env ids fall through
with safe defaults (no normalisation, no discretisation).

The factory sets `env.metadata["action_type"]` to either
`"discrete_original"` or `"discrete_wrapped"` so downstream tools can tell
whether actions came from a native `Discrete` space or from a wrapped `Box`.

## `DiscreteActionWrapper`

Maps a one-dimensional continuous action box onto `n_actions` evenly-spaced
points:

$$
a_i = a_\text{low} + i \cdot \frac{a_\text{high} - a_\text{low}}{n_\text{actions} - 1}, \quad i \in \{0, \dots, n_\text{actions} - 1\}.
$$

- Asserts the underlying space is `Box` with `shape == (1,)`.
- Exposes `action_values` (1-D float array of the mapped values) so tensor
  code can recover the continuous control value for each discrete action.

## `NormalizeObsWrapper`

Linearly rescales observations to $[-1, 1]$ using the environment's declared
bounds:

$$
\tilde s_d = 2 \cdot \frac{\operatorname{clip}(s_d; \, l_d, h_d) - l_d}{h_d - l_d} - 1.
$$

- **Infinite bounds** (common for velocity dimensions, e.g. Pendulum's
  angular velocity) are clipped to ±`_CLIP_FALLBACK = 5.0` *before*
  normalisation. This is deliberately conservative; outliers beyond this
  fallback are clipped to ±1 after scaling.
- `_range` is floor-clipped at `1e-8` to avoid divide-by-zero on degenerate
  dimensions.
- The wrapped `observation_space` reports `low = -1`, `high = +1` of the
  original shape and dtype `float32`.

## Tests

`tests/test_envs.py` covers:

- Registry returns the expected list of env ids.
- `make_env` gives a `Discrete` action space and a `Box` observation space
  on all registered envs.
- `metadata["action_type"]` correctly distinguishes wrapped vs native
  discrete.
- `DiscreteActionWrapper`:
  - action space size matches `n_actions`;
  - endpoints of the linear map are the original `low` and `high`;
  - non-Box action spaces (e.g. CartPole-v1) are rejected with
    `AssertionError`.
- `NormalizeObsWrapper`:
  - every reset observation lies in $[-1, 1]$ up to `1e-5`;
  - the advertised observation space matches;
  - infinite raw bounds (Pendulum) are handled and observations remain
    finite.
