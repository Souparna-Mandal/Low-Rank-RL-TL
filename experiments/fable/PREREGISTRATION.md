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
