# Fable campaign results — low-rank structure in PPO, round 2

Branch: `fable/test`. Protocol: exploration (seeds 0–4, N=5/arm, paired) →
pre-registered confirmation on fresh seeds 100–119 (N=20/arm), identical
hyperparameters, paired one-sided sign-flip permutation tests (20k resamples).
Budgets: CartPole-v1 200 episodes, Acrobot-v1 120 episodes, rollout 1024,
minibatch 128, 6 epochs. See `PREREGISTRATION.md` (committed before launch)
and `docs/research_ideas_explained.md` for the ideas in plain language.

## ✅ CONFIRMED — SSM critic on Acrobot (the primary claim)

`ssm_critic` replaces the MLP critic with a critic that carries an 8-dim
diagonal linear state (h_t = a⊙h_{t−1} + Bφ(s_t), v_t = Ch_t + Dφ(s_t)) —
the low-Hankel-rank prior enforced **by architecture** instead of by penalty.

| Metric (N=20 fresh seeds, paired) | baseline | ssm_critic | Δ | 95% CI | seed wins | p (one-sided) |
|---|---|---|---|---|---|---|
| **AUC** (primary) | −201.4 | −155.9 | **+45.5** | [+31.6, +61.1] | **19/20** | **< 0.0001** |
| Final-quarter mean | −135.9 | −101.0 | **+34.9** | [+24.2, +46.2] | 19/20 | 0.0001 |
| Episodes to solve (roll-10 ≥ −150) | 68.0 (med 66) | 33.8 (med 34) | **2.0× faster** | — | 19/20 | — |

Two qualitative properties beyond the means (see `results/auc_per_seed.png`):
the SSM arm's **worst seed (−171) beats the baseline's mean (−201)**, and its
seed spread collapses (σ ≈ 11 vs 34) — the architecture doesn't just learn
faster, it removes the bad-seed tail.

**Why this one worked when the penalty didn't.** The exploration round also
measured the environments' recurrence structure: Acrobot's fitted global AR(2)
is genuinely second-order (c ≈ [1.51, −0.51]) while CartPole's is ≈ first-order
(c ≈ [1.0, 0.0]). Consistently, the SSM critic was ≈ null on CartPole in
exploration (−0.9 AUC) and large on Acrobot: **the architectural prior pays
off exactly where the value signal actually has the structure** — and, unlike
the HR penalty, it changes the critic's function class (and therefore the
advantages the actor learns from) rather than nudging a scalar output.

## ❌ Did not replicate / null (reported per protocol)

- **ar_explore on CartPole (secondary):** exploration signal (+6.9 AUC, 3/5)
  died on fresh seeds: ΔAUC −1.4, CI [−4.0, +1.0], 9/20 wins, p = 0.85.
  Same fate as every small-N effect in the previous campaign.
- **robust_hd on Acrobot (rider):** ΔAUC +3.3, CI [−4.7, +10.7], p = 0.21 —
  null, though the mechanism itself is validated (it provably preserves a
  25-unit synthetic spike that plain rank-2 Cadzow smears to 1.5, while
  reducing off-spike noise). Worth retesting on LunarLander, where plain
  denoising was actively harmful.
- **latent_ar and ppg_lite (exploration only):** both shared-trunk variants
  collapsed on Acrobot (all seeds at the −500 floor) and underperformed on
  CartPole. Recorded as a negative result for naive shared-trunk PPO at these
  scales; untuned — not a verdict on the ideas at proper hyperparameters.

## Files

- `results/learning_curves.png` — mean return ±95% CI, both envs.
- `results/auc_per_seed.png` — per-seed AUC dot strips with means/CIs.
- `results/stats.json` — all registered numbers.
- `analyze_confirm.py` — reproduces everything from `runs/confirm/`.
- Variants: `src/agents/variants/*.py`; harness: `runner.py`, `launch.py`.

## Next steps suggested by the data

1. **Scale the confirmed result**: ssm_critic on LunarLander (long horizons,
   richer temporal structure) and MountainCar; rank sweep (r ∈ {2, 4, 8, 16});
   compare against a GRU-critic ablation to isolate "linear recurrence" from
   "any recurrence".
2. **Use the AR-order measurement as a cheap predictor** of where the SSM
   critic will pay off (it predicted CartPole-null/Acrobot-win here) — this is
   a thesis-ready diagnostic, and the coefficients are the transfer object.
3. Retune the shared-trunk variants (lower vf_coef, wider trunk) before
   drawing conclusions about ideas 2 and 5.
