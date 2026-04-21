# `low_rank_rl` — package overview

This package is an empirical toolkit for studying **low-rank structure in the
value and policy functions of reinforcement-learning agents**. The central
question it tries to answer on a specific environment / agent pair is:

> *How close is the learned Q-matrix (or its tensor, Hankel, and successor-measure
> reformulations) to a low-rank matrix, and does that structure appear during
> training?*

It is designed so that any agent that implements the `BaseAgent` contract can
be fed into the whole analysis and visualisation stack without modification.

## Package layout

| Sub-package | What lives there |
|---|---|
| `envs/`          | `make_env` factory + `DiscreteActionWrapper`, `NormalizeObsWrapper` |
| `agents/`        | `BaseAgent` + DQN, PPO, tabular Q-learning, SARSA, Monte Carlo |
| `analysis/`      | Rank metrics, HOSVD, Hankel, shifted successor measure |
| `visualization/` | Matplotlib plots for training curves, spectra, value heatmaps |

## Core mathematical objects

### 1. The Q-matrix

For a finite probe set $\{s_1, \dots, s_N\}$ of states and $|\mathcal{A}|$
actions, every agent exposes

$$
Q \in \mathbb{R}^{N \times |\mathcal{A}|}, \qquad Q_{ij} = Q^\pi(s_i, a_j).
$$

The rank of $Q$ is the natural first diagnostic of low-rank structure.

### 2. The value tensor

When the state has a natural product structure
$s = (s^{(1)}, \dots, s^{(k)})$ (e.g. angles and angular velocities in
Acrobot), we discretise each dimension into bins and arrange
$V(s) = \max_a Q(s,a)$ into a tensor

$$
\mathcal{V} \in \mathbb{R}^{n_1 \times \cdots \times n_k}.
$$

Low rank of the **mode-$n$ unfolding**

$$
\mathcal{V}_{(n)} \in \mathbb{R}^{n_n \times \prod_{m \neq n} n_m}
$$

is the central statistic of HOSVD / Tucker analysis (arXiv:2201.09736).

### 3. The Hankel matrix

From a 1-D sequence $f_0, f_1, \dots, f_T$ (values, Q-taken or the policy
along a trajectory) we build

$$
H_{ij} = f_{i+j}, \qquad i \in [0, n), \; j \in [0, T-n+1).
$$

For a sequence driven by a linear dynamical system of order $r$,
$\mathrm{rank}(H) = r$. Low rank therefore detects approximate
finite-dimensional linear (Koopman) dynamics along the policy trajectory
(arXiv:1408.4408).

### 4. The successor measure

For policy $\pi$ and discount $\gamma$,

$$
M^\pi(s, s') = \sum_{t \ge 0} \gamma^t \, \mathbb{P}\!\left(s_t = s' \mid s_0 = s, \pi\right).
$$

Let $\mu^\pi$ denote the stationary distribution of $\pi$. arXiv:2509.05193 shows
that the *shifted* successor measure

$$
\tilde{M}^\pi(s, s') = M^\pi(s, s') - \mu^\pi(s')
$$

has far lower rank than $M^\pi$ in typical environments. We estimate both and
compare their singular-value spectra.

## Rank metrics used throughout

Given the singular values $\sigma_1 \ge \sigma_2 \ge \cdots$ of a matrix $A$:

- **Numerical rank**   $\#\{i : \sigma_i > \tau \sigma_1\}$. Binary, $\tau$-sensitive.
- **Stable rank**      $\mathrm{sr}(A) = \lVert A \rVert_F^2 / \sigma_1^2 = \sum_i \sigma_i^2 / \sigma_1^2$. Scale-invariant, continuous.
- **Effective rank**   $\mathrm{er}(A) = \exp H(\bar p)$ with $\bar p_i = \sigma_i^2 / \sum_j \sigma_j^2$ (Roy & Vetterli 2007). Entropy-based, smooth.

Each captures a different aspect of approximate low-rank structure; we report
all three.

## How things fit together

```
make_env ──► env
                │
                ▼
   build_agent(env) ──► agent (implements BaseAgent.q_matrix)
                │
                ├── training loop ──► durations, reward curves
                │
                └── analysis
                      ├── rank.py       : Q-matrix rank
                      ├── tensor.py     : HOSVD / Tucker on V tensor
                      ├── hankel.py     : Hankel rank + DMD
                      └── successor.py  : M, M̃ and their spectra
                │
                ▼
             visualization.*  (matplotlib)
```

Every `agent` satisfies the `BaseAgent` interface, so new algorithms (e.g. a
low-rank-factored DQN) can be plugged in without touching any analysis
routine.

## References

- Tensor and matrix low-rank value-function approximation in RL — arXiv:2201.09736
- The shifted successor measure — arXiv:2509.05193
- Successor features for transfer — arXiv:2209.14935
- Hankel matrices and Koopman DMD — arXiv:1408.4408
- Roy & Vetterli, *The effective rank*, EUSIPCO 2007
