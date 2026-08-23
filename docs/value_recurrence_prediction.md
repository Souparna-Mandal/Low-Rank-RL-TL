# Predicting future values along a trajectory from their own past

Direct test of the predictive content of the low-Hankel-rank observation. If
Hankel rank ≤ r along (sub-)trajectories, the value signal obeys an order-r
linear recurrence (Kronecker):

```
v_t = c_1·v_{t−1} + c_2·v_{t−2} + … + c_r·v_{t−r}
```

so a small map — r coefficients — fitted from data should forecast future
values along a rollout from just the r most recent ones, without evaluating
the critic at future states. This tests the idea that such a map could supply
cheap value/baseline estimates in policy-optimisation methods.

Code: [`src/analysis/low_rank/recurrence.py`](../src/analysis/low_rank/recurrence.py)
(least-squares AR(r) fit, one-step prediction, recursive free-run), tests in
[`tests/test_recurrence.py`](../tests/test_recurrence.py). Experiments:
`experiments/{dqn_cartpole,dqn_acrobot_2_revised}/exp1_recurrence.ipynb` — 40
greedy trajectories per trained agent (seed 0), global map fitted on 20,
everything evaluated on the 20 held-out. nRMSE = RMSE / RMS of the true
signal; the persistence baseline predicts "same as the previous value".

## Results

### Acrobot (the informative env — the value signal ramps from ≈ −75 to 0)

| test | AR(2) | persistence |
|---|---|---|
| one-step nRMSE, held-out | **0.0173** | 0.0201 |
| free-run 16 steps | 0.046 | 0.040 |
| free-run 64 steps (≈ most of an episode) | **0.097** | 0.267 |
| vs true return-to-go G_t (nRMSE / corr) | **0.50 / 0.77** | critic itself: 0.61 / 0.78 |
| truncated-return completion \|Ĝ−G₀\| | **15.6** | network V(s_T): 15.6 · none: 22.6 |

- A **global** order-2 map transfers across trajectories and extrapolates the
  ramp 64 steps out at ~10% error while persistence degrades to 27% — the
  recurrence captures real dynamics, not just smoothness. r = 2–3 is optimal;
  r = 8 overfits (consistent with the measured Hankel rank).
- The AR(2) extrapolation tracks the **true** discounted return-to-go slightly
  better than the critic itself (it denoises the critic's wobble).
- Completing a truncated return with the AR forecast of V(s_T) costs nothing
  vs evaluating the network at s_T.
- Per-trajectory fits on half an episode extrapolate poorly (nRMSE ≈ 0.32):
  the policy-level global map is the right object.

### CartPole (plateau signal — weak dynamics testbed, strong HR-DQN signal)

The converged value sits on the 1/(1−γ) plateau, so persistence is already at
0.1% one-step error and AR only matches it. The interesting result is the
comparison between agents:

| metric (held-out) | baseline DQN | HR-DQN `progress_order2` |
|---|---|---|
| per-traj fit → free-run 2nd half, median nRMSE | 0.048 | **0.009** |
| critic V vs true G_t, nRMSE / corr | 0.132 / 0.53 | **0.0098 / 0.77** |
| truncated-return completion (AR vs network bootstrap) | 0.538 vs 0.540 | 0.500 vs 0.498 |

The Hankel-regularised agent's value signal is ~5× more self-predictable and
its critic ~13× better calibrated to true returns — the penalty didn't just
lower a rank statistic, it produced a value function that behaves like one.

## Is the speed-up idea right?

**The mechanism is real but the accounting matters.** What the experiment
licenses: given the last r values along a rollout, a fitted global map
predicts future values (and hence baselines/bootstrap targets) about as well
as evaluating the critic at those future states — on a converged policy.

- **Where it genuinely buys something:** (1) *truncated-rollout completion* —
  estimate G₀ from a T-step rollout plus an extrapolated tail, replacing the
  bootstrap network evaluation; equal accuracy measured here. (2) settings
  where value evaluation is *expensive relative to stepping* — big critics,
  ensembles, expensive feature encoders — or where you want a variance-reduced,
  denoised baseline (the AR forecast beat the raw critic against true G_t).
  (3) as a *consistency check/regulariser* (that's HR-DQN itself — the inverse
  direction of the same fact).
- **Where it doesn't:** in standard PG setups (PPO/A2C) the critic shares a
  trunk with the policy, so V comes nearly free with the forward pass you
  already make to act — skipping it saves little. And the map is fitted to the
  *current* policy's value process: mid-training, under policy drift, the
  coefficients lag (the speed campaign's early-engagement failures are the
  same phenomenon seen from the other side).
- **Honest scope:** measured on converged greedy policies, one seed per agent.
  The mid-training version of this claim is untested here.

## Application to classical policy iteration — the state-identity problem

The question: the recurrence predicts *future values along a trajectory*, but
can we know **which state (or state-action pair) those values belong to**, so
they can be placed in the table?

**No — and this is a real wall, not an implementation detail.** The map
c₁…c_r acts on the value *sequence*; it is deliberately blind to state
identity (that blindness is exactly why it is r numbers instead of a model of
the MDP). To know that the value predicted for step t+k belongs to state
s_{t+k}, you must know s_{t+k}, which requires either (a) actually stepping
the environment there — at which point you visited the state and can record
its value normally — or (b) a deterministic known transition model, in which
case you are doing model-based planning and exact evaluation was available
anyway. Under stochastic dynamics it's worse: v̂_{t+k} forecasts the value of
whichever state the *trajectory distribution* delivers — an average over
reachable states — and writing that number into any single table cell is a
category error.

**The rescue is to change what you ask for.** Tabular policy evaluation needs
V(s) only for states you *visit*; future values enter only through the tail
of the return. So use the recurrence where state identity is not needed:

- **Truncated-rollout evaluation:** roll τ steps from s (visited states all
  known), then close with the extrapolated tail Σ_k γ^k v̂_{τ+k} — equivalent
  in the experiment to bootstrapping with the true V(s_τ), and it needs no
  future state identities at all. Shorter rollouts per evaluation pass = the
  actual PI speed-up.
- **Monte-Carlo variance reduction:** replace noisy sampled tails with their
  recurrence forecast (a control-variate flavour of the same trick).

So: the trajectory-prediction part of the idea is correct and measured; the
"place it in the table" part is impossible as stated, but unnecessary — route
the predicted values through return tails instead of table writes.
