# Low-Rank-RL-TL

Empirical study of **low-rank structure in the value and policy functions of
reinforcement-learning agents**. Thesis code for ICL 2026.

Given a trained agent on any Gymnasium environment, the tooling here probes
four complementary views of its internal structure:

1. The **Q-matrix** $Q \in \mathbb{R}^{N \times |\mathcal A|}$ — rank metrics
   on its singular values.
2. The **value tensor** $\mathcal V \in \mathbb{R}^{n_1 \times \cdots \times n_k}$ built from product-structured state spaces — HOSVD / Tucker.
3. The **Hankel matrix** $H_{ij} = f_{i+j}$ of value/Q/policy trajectories — rank + DMD / Koopman.
4. The (shifted) **successor measure** $M^\pi$ and $\tilde M^\pi = M^\pi - \mathbf 1 \mu^\top$ — arXiv:2509.05193.

Each view is backed by a `BaseAgent` contract so new algorithms plug in
without touching any analysis code.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
```

## Quick tour

```python
from low_rank_rl.envs   import make_env
from low_rank_rl.agents import DQNAgent
from low_rank_rl.analysis.rank   import compute_rank_metrics, sample_states
from low_rank_rl.analysis.tensor import build_value_tensor, hosvd_spectra
from low_rank_rl.visualization   import plot_singular_value_spectrum, plot_hosvd_spectra

env    = make_env("Acrobot-v1")                      # raw 6-D obs discretised [7]*6
agent  = DQNAgent(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n)
# ... train ...

states  = sample_states(env, 500)                     # full canonical grid — n ignored
metrics = compute_rank_metrics(agent, states)
print(metrics.summary())
plot_singular_value_spectrum(metrics, save_path="q_spectrum.png")

V_tensor = build_value_tensor(agent, env, dims=[0, 1, 2, 3, 4, 5])
plot_hosvd_spectra(hosvd_spectra(V_tensor), save_path="hosvd.png")
```

## End-to-end experiments

```
python experiments/run.py --config experiments/configs/acrobot_dqn.yaml
```

See `experiments/README.md` for the config schema and the full list of
artefacts that the runner produces.

## Repository layout

```
low_rank_rl/
├── agents/         # BaseAgent + DQN, PPO, Q-learning, SARSA, Monte Carlo
├── analysis/       # rank, HOSVD, Hankel, successor-measure
├── envs/           # Gymnasium factory + action/obs wrappers
├── visualization/  # matplotlib plots (training, spectra, heatmaps)
└── README.md       # package overview + maths

experiments/        # YAML configs + run.py
tests/              # pytest suite (pytest -q)
papers/             # reference papers
```

Every important folder ships its own `README.md` with the specific maths and
design notes for that layer.

## Testing

```
pytest -q
```

The suite is fully headless (`matplotlib` Agg backend) and runs in a few
seconds. See `tests/README.md` for what each module covers.
