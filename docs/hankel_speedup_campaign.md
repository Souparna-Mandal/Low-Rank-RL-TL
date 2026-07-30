# HR-DQN speed campaign log

Goal: make the Hankel regulariser produce **faster training** than classical DQN
(baseline CartPole solve episodes [608, 545, 841, 557], mean 638). Metrics per run:
`solve-ep` (first episode whose rolling-50 mean crosses 475 — the loop's early-stop
rule), `AUC600` (mean reward over the first 600 episodes, solved runs padded at 500),
`eval20` guardrail (final 20 greedy episodes; must not degrade).

One branch per feature set, one round per branch, 4 seeds per variant, CartPole first.

## Round 1 — branch `dqn/hankel-regularisation-speedup-1`

Hypothesis: forensics showed replay windows are near-manifold from the start
(rel-tail ≈ 2%), so the warm-up wastes the early-shaping window — engage λ from
step 0 (safety gate ρ=0.25 on) and vary strength / schedule / rank / signal / width.
New agent features: `decay_grad_steps` (λ decays linearly to 0), `hankel_signal: v`
(penalise max_a Q along windows).

| variant | solve-ep (4 seeds) | mean | AUC600 | eval20 |
|---|---|---|---|---|
| baseline | [608, 545, 841, 557] | 638 | 163.5 | 490.0 ± 14.2 |
| tail_lo (ref) | [413, 753, 875, 704] | 686 | 175.6 | 500.0 ± 0.0 |
| gated (ref) | [798, 1364, 419, 381] | 740 | 177.9 | 494.5 ± 8.0 |
| early_hi | [460, 1500, 801, 685] | 862 | 162.0 | 393.6 ± 184.2 |
| early_strong | [514, 1500, 796, 730] | 885 | 176.3 | 386.8 ± 195.7 |
| decay_strong | [642, 1500, 1033, 761] | 984 | 145.7 | 378.2 ± 211.0 |
| order1_hi | [862, 1006, 555, **361**] | 696 | **187.6** | 458.2 ± 72.5 |
| vseq_hi | [686, 1500, 1500, 698] | 1096 | **197.3** | 269.7 ± 231.1 |
| w32_hi | [698, 798, 1323, 770] | 897 | 158.6 | 495.1 ± 8.5 |

**Verdict: hypothesis falsified.** Engaging the penalty from step 0 destabilises a
seed in most variants (1500-cap runs with collapsed eval). Early Q is low-rank but
*wrong* — regularising it then consolidates flat-but-bad value structure before TD
has shaped anything; the warm-up in the original `gated`/`tail_lo` was load-bearing.
Two useful signals survive: (1) `order1_hi` produced the campaign's fastest solve
(361) and the second-best AUC — target rank 1 pulls harder in the right direction
*when it works*; (2) `vseq_hi` has the best AUC of all — V-smoothing accelerates the
early/mid phase — but collapses late (over-smoothed V destabilises the greedy
policy). Fixed clocks mistime per-seed; engagement should follow learning progress.

## Round 2 — branch `dqn/hankel-regularisation-speedup-2` (running)

Hypothesis: aggressive pulls on the *proven* schedule + adaptive (TD-conditioned)
engagement instead of a wall clock. New feature: `td_gate_scale` — a per-window
TD-consistency gate (penalise only windows whose own bootstrap residual is ≤
scale × current batch TD loss, i.e. regularise where TD already locally agrees).

Variants: `gated_order1` (r=1, λ=1e-2, warm-up 2000 + ramp 2000, ρ=0.25),
`gated_strong` (λ=5e-2, same schedule), `vseq_decay` (signal=v, λ=1e-2, warm-up
1000, decay 6000 — accelerate then release), `tdgate_hi` (λ=1e-2, no warm-up,
TD-consistency gate scale 1.0).

| variant | solve-ep (4 seeds) | mean | AUC600 | eval20 |
|---|---|---|---|---|
| baseline | [608, 545, 841, 557] | 638 | 163.5 | 490.0 ± 14.2 |
| **gated_order1** | **[540, 734, 500, 662]** | **609** | **191.2** | **493.9 ± 10.5** |
| gated_strong | [700, 717, 484, 1500] | 850 | 173.9 | 371.1 ± 206.0 |
| vseq_decay | [526, 1500, 1500, 715] | 1060 | 170.3 | 279.6 ± 223.1 |
| tdgate_hi | [859, 735, 1500, 1500] | 1148 | 144.9 | 288.6 ± 215.1 |

**Verdict: first clean win.** `gated_order1` — target rank 1 on the proven warm-up+
ramp schedule — beats baseline on *every* metric with no pathological seed: −29
episodes mean solve, tighter worst case (734 vs 841), +17% AUC600, better final
eval. λ=5e-2 is too strong even mid-schedule; the V-signal keeps collapsing seeds;
the TD-consistency gate misfires when engaged from step 0 (same early-engagement
disease). Rank-1 reading: at convergence the on-policy value sequence saturates at
energy-rank 1, so r=1 pulls toward the *actual* solution manifold rather than the
looser r=2 shell — a harder, better-aimed pull once TD has shaped Q.

## Round 3 — branch `dqn/hankel-regularisation-speedup-2` (replication + tuning, no new code)

N=10 seeds for `baseline` and `gated_order1` (seeds 0–9), plus local tuning at 4
seeds each: `gated_order1_l2` (λ=2e-2) and `gated_order1_early` (warm-up 1000 +
ramp 1000 — engage the rank-1 pull one phase earlier).

| variant | N | solve mean±sd | median | AUC600 | eval20 |
|---|---|---|---|---|---|
| baseline | 10 | 938 ± 342 | 874 | 138.1 | 457.2 ± 115.4 |
| gated_order1 | 10 | 937 ± 385 | 800 | 165.8 | 364.1 ± 203.3 |
| gated_order1_l2 | 4 | 826 ± 360 | 670 | 166.6 | 488.3 ± 20.3 |
| gated_order1_early | 4 | 873 ± 370 | 728 | 183.8 | 394.5 ± 175.4 |

Permutation tests (N=10 vs N=10): solve-ep diff ≈ 0 (p = 0.50); AUC600 +27.8
(p = 0.081, suggestive only).

**Verdict: the round-2 win does not replicate at N=10.** Seeds 4–9 are much harder
for everyone (baseline mean jumps 638 → 938), and `gated_order1` bifurcates: it
accelerates the easy seeds (500/540/662/694/734) but hits the 1500 cap with eval
collapse on three hard seeds. Diagnosis: **clock-based engagement is the bug** — a
fixed grad-step warm-up fires while slow seeds' Q is still wrong, and the rank-1
pull then consolidates bad structure (the same disease as round 1, one phase
later). Engagement must be conditioned on each run's own learning progress, not on
a wall clock.

## Round 4 — branch `dqn/hankel-regularisation-speedup-3` (progress-conditioned engagement)

New feature: `engage_reward_threshold` / `engage_reward_window` — the agent tracks
its own rolling episode returns (from the episodic buffer) and latches the penalty
on only once the rolling mean crosses the threshold; the ramp then runs from the
engagement point. Slow seeds engage late, fast seeds early — per-seed timing for
free, no training-loop changes.

Variants (seeds 0–9 for power, vs cached N=10 baseline): `progress_order1`
(r=1, λ=1e-2, engage@100/10eps, ramp 2000, ρ=0.25), `progress_order1_l2` (λ=2e-2),
`progress_order2` (r=2 control for the rank choice).
