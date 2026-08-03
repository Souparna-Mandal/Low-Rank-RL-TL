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
