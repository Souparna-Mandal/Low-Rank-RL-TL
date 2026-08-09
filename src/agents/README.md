# Agents

Two value-based agents live here, both driven by the same shared training loop
([`src/training.py`](../training.py)) and the same config-driven wiring
([`src/experiment.py`](../experiment.py)):

| Agent | File | Role |
|---|---|---|
| `QAgent` | [`q_agent.py`](q_agent.py) | The DQN baseline — uniform replay, 1-step TD, optional Double DQN. |
| `RainbowDQNAgent` | [`rainbow_agent.py`](rainbow_agent.py) | The **sample-efficiency benchmark** — a Rainbow agent (IQN variant) to measure the low-rank Hankel technique against. |

This document is the manual for `RainbowDQNAgent`: what it is, the one design idea
that keeps it clean, how to plug it in, and every knob you can turn.

---

## The shared agent contract

The training loop never imports a specific agent — it only calls a fixed set of
methods. Any agent that implements these drops straight in:

| Method / attribute | Called by the loop to… |
|---|---|
| `pi(state)` | pick an action for the current step |
| `update_buffer(...)` / `update_buffer_atari(...)` | store the transition just observed |
| `train()` | run gradient updates from replayed experience |
| `update_target_network()` | pull the target network toward the online one |
| `decay_epsilon()` | anneal exploration (a no-op for Rainbow — see below) |
| `save(path)` / `load(path)` | checkpoint / restore |
| `replay_buffer`, `epsilon` | length / status for logging + DEBUG prints |

`RainbowDQNAgent` **reuses** `QAgent`'s `save`, `load`, `update_target_network`
and `act_greedy`, and overrides only the pieces Rainbow actually changes. That is
why the training loop, the checkpointing, and the low-rank analysis all keep
working without a single edit outside this file.

---

## What Rainbow is

"Rainbow" (Hessel et al., 2018) is DQN with six independent improvements stacked
together, each fixing a different weakness. Our variant swaps the original C51
distributional head for **IQN** (implicit quantile networks), which is stronger
and removes the awkward `v_min`/`v_max` value-range tuning.

| Ingredient | What it fixes | Where it lives |
|---|---|---|
| **Double DQN** | overestimation bias in the bootstrap target | `double` flag (online net selects the next action, target net evaluates it) |
| **Dueling** | wasted capacity when actions barely matter in a state | value + advantage streams in the head, combined per quantile |
| **Prioritised replay (PER)** | learning equally from boring and surprising transitions | `PrioritizedReplayBuffer` + `SumTree` |
| **Multi-step returns** | slow reward propagation from 1-step bootstrapping | n-step accumulator in `_ingest` / `_emit_front` |
| **Distributional (IQN)** | throwing away everything but the mean of the return | cosine-τ embedding + quantile-Huber regression |
| **Noisy Nets** | clumsy, un-annealed ε-greedy exploration | `NoisyLinear` layers in the head |

---

## The one design idea: encoder in, algorithm head in the agent

The other experiments pass a **whole Q-network** into the agent (e.g. `NatureCNN`).
Doing that for Rainbow would force every experiment's network to hand-implement
noisy layers, dueling streams, and the IQN quantile machinery — a heavy, error-prone
contract to re-satisfy per game.

Instead, `RainbowDQNAgent` splits the network in two:

```
   your experiment supplies                   this module supplies
 ┌───────────────────────────┐   features   ┌────────────────────────────────┐
 │   encoder  (perception)   │  ──────────▶ │  RainbowIQNNetwork  (the head) │
 │   obs → (B, feature_dim)  │   (B, F)     │  τ-embedding · dueling · noisy │
 └───────────────────────────┘              └────────────────────────────────┘
        game-specific                              reusable, algorithm-specific
```

**You write only the encoder** — the perception part that changes per environment.
The Rainbow head (noisy + dueling + IQN) is written once, here, and shared by every
experiment. The agent even derives the head's dimensions for you: `feature_dim` from
a dummy forward through your encoder, and `n_actions` from the env.

### The encoder contract (all of it)

```python
encoder(obs) -> Tensor of shape (B, feature_dim)
```

That is the entire requirement. The encoder owns its own input normalisation — for
Atari that means casting `uint8 → float / 255` inside `forward`, exactly as
`NatureCNN` already does. No knowledge of quantiles, noise, or actions is needed.

---

## Using it

### 1. Write an encoder

The Nature-DQN convolutional **body**, with the head removed:

```python
import torch.nn as nn

class NatureEncoder(nn.Module):
    """(C, 84, 84) uint8 frame stack -> flat feature vector."""
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x.float() / 255.0)   # owns its normalisation
```

### 2. Wire it through `build_agent`

`build_agent` takes an optional `agent_cls` (defaults to `QAgent`). Pass the
Rainbow class and your encoder:

```python
from agents.rainbow_agent import RainbowDQNAgent

agent = build_agent(
    cfg, env,
    NatureEncoder,
    {"in_channels": obs_shape[0]},     # nn_extra_kwargs: encoder args ONLY
    agent_cls=RainbowDQNAgent,
)
```

Note what is **not** in `nn_extra_kwargs`: no `n_actions`, no `fc_hidden`. Those
belong to the head and the agent supplies them itself.

### 3. Configure the `agent` block

The keys under `agent:` in `config.yaml` are passed straight into the constructor,
so they must match `RainbowDQNAgent.__init__`. A sensible Atari starting point:

```yaml
agent:
  replay_buffer_capacity: 250000
  batch_size: 32
  nn_learning_rate: 0.00005
  discount_factor: 0.99
  n_step: 3
  n_quantiles: 32          # online quantile samples N   (loss)
  n_quantiles_target: 32   # target quantile samples N'  (loss)
  n_quantiles_act: 32      # quantiles averaged for the greedy / analysis Q
  n_cos: 64                # cosine basis size for the τ embedding
  head_hidden: 512
  huber_kappa: 1.0
  noisy_sigma0: 0.5
  dueling: true
  per_alpha: 0.5           # prioritisation strength (0 = uniform replay)
  per_beta_start: 0.4      # importance-sampling correction, annealed to 1
  per_beta_increment: 0.0000004
  per_eps: 0.00001
  TD_LR: 0.33              # soft target-update rate (Polyak), reused from QAgent
  buffer_util: 4
  gd_steps_ceil: 100
  grad_clip_norm: 10.0
  double: true
```

`device`, `q_network` (the encoder), `nn_extra_kwargs`, and `env` are injected by
`build_agent` — they are code/experiment objects, not config keys.

---

## Configuration reference

**Replay & optimisation**

| Key | Meaning |
|---|---|
| `replay_buffer_capacity` | max transitions kept in the prioritised buffer |
| `batch_size` | transitions per gradient step |
| `nn_learning_rate` | AdamW learning rate (amsgrad, as in `QAgent`) |
| `discount_factor` | γ for the (multi-step) return |
| `buffer_util` | throttles updates: `gd_steps = len(buffer) // (buffer_util · batch_size)` |
| `gd_steps_ceil` | hard cap on gradient steps per `train()` call |
| `grad_clip_norm` | global-L2 gradient clip |
| `TD_LR` | soft target-update rate: `target += TD_LR·(online − target)` |

**Multi-step**

| Key | Meaning |
|---|---|
| `n_step` | horizon of the n-step return (`1` recovers standard 1-step TD) |

**Distributional (IQN)**

| Key | Meaning |
|---|---|
| `n_quantiles` | number of online quantile samples in the loss (N) |
| `n_quantiles_target` | number of target quantile samples in the loss (N′) |
| `n_quantiles_act` | quantiles averaged into the expected Q used for acting, Double-DQN selection, and the Hankel traces |
| `n_cos` | size of the cosine basis used to embed each quantile fraction τ |
| `head_hidden` | width of the value/advantage hidden layers |
| `huber_kappa` | Huber threshold κ in the quantile-Huber loss |

**Prioritised replay**

| Key | Meaning |
|---|---|
| `per_alpha` | how strongly priority skews sampling (`0` → uniform) |
| `per_beta_start` | starting importance-sampling exponent (bias correction) |
| `per_beta_increment` | per-sample step by which β climbs toward 1.0 |
| `per_eps` | floor added to every priority so nothing is unsamplable |

**Architecture / algorithm switches**

| Key | Meaning |
|---|---|
| `dueling` | enable the value/advantage decomposition |
| `noisy_sigma0` | initial noise scale σ₀ in every `NoisyLinear` |
| `double` | Double-DQN next-action selection (recommended `true`) |

---

## How one training step works

Each gradient step inside `train()`:

1. **Resample noise** on both networks (`reset_noise()`), so exploration and the
   target both see fresh weight perturbations.
2. **Sample a prioritised batch.** The sum tree draws transitions in proportion to
   their last error, and returns importance-sampling weights that scale down the
   loss of over-sampled transitions.
3. **Build the target quantiles.** For each transition the stored **n-step return**
   is `r = Σ γ^k r_k`, and the bootstrap is `γ^n · Z(s', a*)` where `a*` is chosen by
   the **online** net (Double DQN) and its distribution `Z` read from the **target**
   net at freshly sampled quantile fractions. Terminal transitions drop the bootstrap.
4. **Regress with the quantile-Huber loss.** Online quantiles for the taken action
   are compared against the target quantiles over all pairs `(τ_i, τ'_j)`; the
   asymmetric weight `|τ_i − 1{δ<0}|` is what makes each output head converge to its
   quantile of the return distribution.
5. **Feed the error back as priority.** The per-sample loss becomes each
   transition's new priority, so surprising transitions get replayed more often.

`forward()` on the network returns the **mean of the quantiles** — an ordinary
expected-value Q — so greedy action selection (`act_greedy`) and everything
downstream stay exactly as they were for `QAgent`.

---

## Compatibility with the low-rank Hankel analysis

The whole point of this agent is to be analysed the same way as the DQN baseline,
so it deliberately preserves the analysis contract in
[`analysis/low_rank/hankel_policy.py`](../analysis/low_rank/hankel_policy.py):

- `policy_net(state)` → **expected Q** `(B, n_actions)` → drives the **Hankel Q** trace.
- `policy_net.value_advantage(state)` → **(V, A)** from the dueling streams → drives
  the **Hankel V** and **Hankel A** traces.

Because Rainbow is genuinely dueling, the **advantage trace is now non-trivial** —
unlike the vanilla DQN net, where the config notes `advantage trace is trivial`.
So this benchmark unlocks a real Hankel-A signal for the low-rank study.

---

## Notes & gotchas

- **Exploration is noisy-only.** There is no ε-greedy. The `EpsilonGreedyExplorer`
  mixin is initialised inert (`ε ≡ 0`) purely so the shared loop's `decay_epsilon()`
  call and DEBUG `agent.epsilon` read keep working. Tune exploration via
  `noisy_sigma0`, not an ε schedule.
- **Scale unclipped Atari rewards.** Canonical Rainbow clips Atari scores to [-1, 1];
  raw scores (Seaquest: 20..1000+) make the PER priorities heavy-tailed and crush the
  max-normalised IS weights, silently shrinking the effective learning rate. The fix
  is environment-level, not an agent knob: set `reward: {scale: 0.01}` in the config's
  `environment` block ([`ScaleReward`](../environments/wrappers/reward_wrappers.py)).
  The wrapper keeps the raw score in `info["raw_reward"]`, which the training loop
  logs — so reward curves and `solved_reward` stay in raw game units.
- **Target updates are soft, not hard.** We reuse `QAgent.update_target_network`
  (a Polyak step controlled by `TD_LR`), matching the rest of the codebase rather
  than the hard periodic copy in the original Rainbow paper.
- **IQN adds sampling noise to the Hankel traces.** `forward()` samples fresh τ
  every call, so the Q/V/A sequences carry a little quantile-sampling variance.
  Raising `n_quantiles_act` shrinks it; if you need perfectly deterministic traces
  for the low-rank study, ask for an eval path (fixed midpoint τ + noise disabled).
- **Multi-step and truncation.** The n-step accumulator is flushed on `terminated`,
  which is the only end-of-episode signal the training loop passes to `update_buffer`.
  On **truncation** (time-limit cutoffs) the loop does not signal the agent, so a
  window can briefly span the episode boundary. This mirrors the existing baseline's
  own handling; flag it if your env truncates often and you want it fixed.

---

## Quick verification

A standalone end-to-end check (CartPole, real training loop, all six ingredients
exercised) lives outside the repo tree while iterating; the agent should climb from
~random to a clearly higher return within ~120 episodes and collect Hankel Q/V/A
sequences without error. Use it as the template when wiring a new encoder.
