# MountainCar SB3 tuning log

Goal: recover FHR-DQN's sample-efficiency edge over the stock SB3 baseline on
MountainCar-v0, with the baseline faithful to the published RL-Zoo recipe.
"Sample efficiency" here = how early learning starts and how fast it
converges: env steps to greedy-eval reward thresholds (-160/-140/-120/-110)
and eval reward at fixed step budgets — not just the final plateau.

Every iteration: hypothesis -> one config change -> baseline + FHR arms x 3
seeds -> verdict. Infrastructure: `make_variants.py` (variant dirs),
`launch_variants.py` (fan-out), `summarize.py` (metrics tables).

## Iteration 0 — diagnosis (no new runs)

Facts established from the existing `mountaincar/` zoo-config runs (seeds
44/66, arms baseline + exp1-9) and the classic
`experiments/classical_control/dqn_mountaincar` runs:

1. **The classic implementation was never sample-efficient in absolute
   terms.** Classic baseline first reached rolling(10) train reward -160 at
   ~350k env steps (seed 44: never, 400k budget); the classic FHR winner at
   ~158-194k. The SB3 zoo baseline reaches -160 at ~65-78k steps — already
   2-4x faster than the classic FHR winner. FHR's old 3/3-seed edge lived in
   a slow-learning regime (lr 5e-4, Polyak tau 0.005, buffer 100k, gamma
   0.99, normalised obs, 1 grad step / 2 env steps); the RL-Zoo recipe (lr
   4e-3, hard target sync / 600 steps, buffer 10k, gamma 0.98, raw obs, 8
   grad steps / 16 env steps) removed most of that headroom.
2. **The FHR penalty is mechanically healthy under the zoo recipe** (not a
   port bug): b_h ~ 126/128 batch rows carry valid lag chains, coefficients
   sit at the Bellman-consistent fixed point (sum_c ~ 1, companion radius
   ~ 1), penalty_weighted ~ 10-30% of TD loss. It fires; it just doesn't
   produce a consistent edge — none of exp1-9 (lambda 0.01-0.5, r 2-4, ARX)
   beat baseline consistently on both seeds.
3. **The apparent "worse than published baseline" is (at least partly) an
   evaluation-protocol artifact.** RL-Zoo reports -100.85 +/- 9.9 for DQN
   MountainCar-v0 with *deterministic* eval episodes; the notebooks read
   eps-greedy *training* rewards with exploration_final_eps 0.07, which on
   MountainCar reads ~20-30 lower and much noisier. Fixed by the new
   GreedyEvalCallback (eval.csv: 10 deterministic episodes on fixed reset
   seeds every 5k steps, zero global-RNG contamination — the training stream
   is bit-identical with and without it).
4. Config deviations from the published zoo entry: n_timesteps 150000 vs
   zoo's 120000 (stretches the eps decay from 24k to 30k steps since
   exploration_fraction is relative), plus the repo's early-stop gate which
   zoo doesn't have. Everything else matches hyperparams/dqn.yml exactly.
5. Two-seed comparisons are underpowered for this env (baseline final25
   spread: -129 vs -187 across seeds 44/66). Tuning runs use 3 seeds
   (44, 66, 52); winners should be confirmed on 5.

## Phase A — update-cadence granularity (in progress)

Hypothesis (user): at a fixed 0.5 gradient-steps-per-env-step ratio, finer
bursts (fewer env steps between updates) start learning earlier — and matter
more for FHR, whose penalty/ramp bookkeeping was designed against the classic
loop's 1-grad-step-per-2-env-steps cadence.

Variants (everything else = zoo recipe, analysis ticks off, greedy eval on):

| variant  | train_freq | gradient_steps |
|----------|-----------:|---------------:|
| cad64_32 | 64         | 32             |
| cad16_8  | 16         | 8 (zoo ref)    |
| cad4_2   | 4          | 2              |
| cad2_1   | 2          | 1              |

Arms per variant: baseline (lambda 0), exp1 (lambda 0.5, r=2, c_lr 0.03 — the
classic 3/3 winner block), exp2 (lambda 0.1, r=2). Seeds 44/66/52.

Results: pending.
