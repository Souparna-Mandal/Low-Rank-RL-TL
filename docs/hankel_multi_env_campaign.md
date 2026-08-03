# HR-DQN multi-env sample-efficiency campaign

Goal: establish the best sample efficiency the Hankel-rank structure can buy
on a broadened classical-control suite, extending the CartPole/Acrobot results
(docs/hankel_speedup_campaign.md). Branch: `dqn/hankel-multi-env-efficiency-1`
(local). All runs CPU-pinned (tiny MLPs: faster than MPS, fully parallel).

## New environments

| dir | env | obs → act | notes | solved |
|---|---|---|---|---|
| dqn_mountaincar | MountainCar-v0 | 2 → 3 | sparse reward, hard exploration | ≥ −110 |
| dqn_pendulum | Pendulum-v1 | 3 → 9 | torque discretised (9 bins), never terminates | ≥ −200 |
| dqn_lunarlander | LunarLander-v3 | 8 → 4 | Box2D | ≥ 200 |
| dqn_cliffwalking | CliffWalking-v1 | 48 (one-hot) → 4 | 200-step cap added | ≥ −25 |
| dqn_frozenlake | FrozenLake8x8-v1 | 64 (one-hot) → 4 | slippery (stochastic) | ≥ 0.70 |

New plumbing (committed): `OneHotObservationWrapper`, `DiscretiseActionWrapper`,
`time_limit` key in `make_environment`.

## Protocol

Per env: `solve-ep` (first episode with rolling-50 mean ≥ solved), `AUC`
(mean training return over the episode budget, solved runs padded), final
20-greedy-episode eval as guardrail. Variants follow the established recipe
space: `baseline` (λ=0), constant-λ `tail_lo/hi`, and the progress-latch
`progress(_lo)` with per-env engagement thresholds. Campaign law from the
CartPole/Acrobot rounds: measure the on-policy Hankel rank per env before
choosing `hankel_order` — never penalise below the measured rank.

## Round 0 — baseline sanity (2 seeds/env)

(running)

(round 0: all five baselines dead under per-episode training — retuned to
mid-episode cadence. Round 0b: Pendulum/LunarLander learn; CliffWalking/
FrozenLake weak → 0c tunes; MountainCar flatlined twice → excluded,
exploration-bound.)

## Round 1 (variant grids on healthy envs + 0c probes)

| env | takeaway |
|---|---|
| Pendulum | grid flat at N=4 (AUC spread 2.7%, eval σ≈150) — baseline never reaches −200 in 800 eps; needs denser training before variants can differentiate |
| LunarLander | tail_lo +11 AUC over baseline (−40.8 vs −51.8), eval intact — directional, confirmation launched |
| CliffWalking (0c) | bimodal: 1/2 seeds learns (eval −107.5 ± 92.5) — grid launched to test whether the regulariser moves the flakiness |
| FrozenLake (0c) | γ=0.95 + denser training: seed 0 SOLVES (ep 2202), eval 0.57 — grid launched |

## Round 2 (launched): FrozenLake grid, CliffWalking grid, LunarLander
tail_lo confirmation (seeds 4–9), Pendulum train_freq 8 + lr 5e-4 probe.

## FrozenLake stochasticity contrast (committed before CliffWalking grid lands)

Slippery (stochastic): every Hankel variant harms — baseline AUC 0.25 vs
tail_hi 0.03. Deterministic twin (`dqn_frozenlake_det`, is_slippery=false),
N=4: **parity** — baseline 0.17, tail_lo 0.17, tail_hi 0.14. Stochastic
transitions are isolated as the factor that invalidates the trajectory-rank
premise (a realized value path under slips is a Markov sample path, not a
near-recurrence).

**Metric rule for tabular envs, stated before the CliffWalking grid returns:**
primary = training AUC. Greedy eval20 on deterministic gridworlds is
argmax-cycle-fragile (a greedy loop hits the step cap and scores 0/−200
despite ≥90% ε-greedy training success) and is recorded as observation only.
Noted: on frozenlake_det the Hankel variants produce cycle-free greedy
policies more often than baseline (2/4 vs 0/4) — possible structure effect,
not a claim.
