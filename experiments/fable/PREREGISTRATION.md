# Fable campaign — confirmation round pre-registration

Registered BEFORE launching the confirmation runs. Exploration (seeds 0–4,
N=5/arm, paired) found: ssm_critic +37.9 AUC on Acrobot (5/5 seed wins on both
metrics); ar_explore +6.9 AUC / +17.9 final-quarter on CartPole (3/5);
robust_hd +4.2/+9.8 on Acrobot (3/5, 4/5). latent_ar and ppg_lite collapsed on
Acrobot (all seeds at −500 floor) and underperformed on CartPole — recorded as
negative exploration results, not carried forward.

## Registered confirmations (fresh seeds 100–119, N=20/arm)

Budgets and hyperparameters identical to exploration (CartPole 200 episodes,
Acrobot 120 episodes, rollout 1024, minibatch 128, 6 epochs, defaults
otherwise). All arms share the same seed list (paired permutation tests on
per-seed differences, 20,000 resamples, one-sided in the direction found in
exploration, alpha 0.05).

1. **PRIMARY — ssm_critic vs baseline on Acrobot-v1.**
   Primary metric: AUC (mean return per episode over the 120-episode budget).
   Secondary: final-quarter mean return.
2. **SECONDARY — ar_explore vs baseline on CartPole-v1.**
   Primary metric: AUC. Secondary: final-quarter mean.
3. **RIDER — robust_hd vs baseline on Acrobot-v1.**
   Same metrics; weakest exploration signal, reported either way.

Anything that fails here is reported as "did not replicate". No hyperparameter
changes between exploration and confirmation for these arms.

---

## Addendum 2 — registered before launch (round 2)

Probe results (seeds 0–4): LunarLander-v3 ssm_critic vs baseline +23.9 AUC,
5/5 paired wins, final-quarter +55.3. Acrobot rank sweep: effect robust for
rank ∈ {2,4,8,16} (AUC −167…−152 vs baseline −199).

**CONFIRMATION 2 — ssm_critic vs baseline on LunarLander-v3.** Fresh seeds
100–119 (N=20/arm), 150 episodes, hyperparameters identical to the probe
(defaults, rank 8). Primary: AUC, one-sided paired sign-flip permutation,
alpha .05. Secondary: final-quarter mean. Failure is reported as
non-replication.

Exploration round 2 (seeds 0–4, NOT claims): {baseline, ssm_critic, ssm_auto}
on Pendulum-v1 / MountainCar-v0 / CliffWalking-v1; ssm_auto on CartPole +
Acrobot; gru_critic ablation on Acrobot.

---

## Addendum 3 — registered before launch (round 3: ssm_auto)

Exploration (fixed code, seeds 0–4): ssm_auto beats fixed-rank ssm_critic on
BOTH original envs (CartPole AUC 59.7 vs 53.7 vs baseline 54.6; Acrobot −152.8
vs −160.9 vs −198.8), self-selecting orders 2–3 on CartPole and 2–5 on
Acrobot. Orders on other envs: Pendulum 3–4, CliffWalking 2–3, MountainCar
3–5 (all null envs).

**CONFIRMATION 3a (PRIMARY) — ssm_auto vs baseline on CartPole-v1.** Fresh
seeds 100–119 (N=20/arm; baseline cached from confirmation 1), 200 episodes,
defaults. Primary: AUC, one-sided paired permutation, alpha .05. This is the
sharpest test of the AR-order trick: fixed-rank SSM was null here.
**CONFIRMATION 3b (SECONDARY) — ssm_auto vs ssm_critic on Acrobot-v1.** Fresh
seeds 100–119 (ssm_critic cached from confirmation 1), 120 episodes. Metric:
AUC, one-sided (H1: adaptive > fixed), alpha .05.
Rider (context, no claim): ssm_critic on CartPole fresh seeds 100–119.

---

## Addendum 4 — registered before launch (round 4)

**CONFIRMATION 4a — ssm_critic vs baseline on LunarLander-v3, LONG BUDGET.**
Motivated by round 2's significant registered secondary (final-quarter +18.1,
p=.018) with a null primary at 150 episodes. Budget doubled to 300 episodes,
fresh seeds 200–219 (N=20/arm), defaults (rank 8). **PRIMARY (pre-declared
switch): final-quarter mean return**, one-sided paired sign-flip permutation,
alpha .05. Secondary: AUC. Failure on the primary is reported as a definitive
non-replication of the LunarLander hypothesis.

**CONFIRMATION 4b — ssm_auto vs baseline on CartPole-v1 at N=40.** Round 3a
was directionally positive (p=.086) at N=20. Adding seeds 120–139 (N=20 new
per arm), pooled with 100–119 → N=40, pooling declared here in advance.
Primary: AUC pooled N=40, one-sided, alpha .05. Sensitivity: seeds 120–139
alone (reported either way).

**4c — AR(2) coefficient transfer probe (measurement, no performance claim).**
Fit AR(2) to each rollout's value sequences during baseline PPO training
(after update 5) on CartPole / Acrobot / LunarLander, 10 seeds each. Report:
within-env cross-seed coefficient dispersion vs cross-env separation
(are the coefficients an environment property — the transferable object?).
