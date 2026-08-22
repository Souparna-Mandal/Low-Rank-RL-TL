# Research TODO — FHR / Hankel low-rank RL: experiments & figures

Working plan for the results we want to *show*, organised as six tracks. Each
track lists the experiments to run and the figures they feed. Checkboxes track
status; items marked **[have]** already exist in the repo and only need a
figure pass.

Conventions for every experiment below: ≥3 paired seeds (5 where cheap), the
baseline and FHR arm differ **only** in `fhr_weight` (and its FHR block),
identical gradient-step budgets, greedy/near-greedy eval protocol as already
encoded per stack. Rank measures quoted everywhere: `energy_rank` at 0.999 +
effective rank, from `src/analysis/low_rank/rank.py` — one definition, used in
every figure.

---

## Track 1 — Motivation: value sequences along trajectories are low-Hankel-rank, everywhere

Purely observational, no method changes: fit/measure Hankel structure of
Q(s_t, a_t) (or V(s_t)) along on-policy trajectories of *trained* agents
across every domain we can reach. This is the opening figure of the story.

### Experiments
- [ ] **Classical control** (CartPole, MountainCar, Acrobot, LunarLander):
      Hankel spectra + AR-probe on trained DQN/FHRDQN baselines.
      **[have]** seams: `collect_hankel_sequences`, `autoregressive_value_probe`,
      per-episode spectra already logged by the run stack; needs a consolidation
      notebook, not new runs.
- [ ] **Atari-100k** (Krull now; suite games as runs land): Hankel rank of the
      IQN mean-Q along greedy trajectories for BBF / EfficientRainbow
      baselines. **[have]** the trajectory traces in the run dirs.
- [ ] **Longer Atari runs**: rerun the analysis pass on whatever >100k-step
      runs exist (Pacman/Seaquest DER-style runs) — shows the structure is not
      a small-data artefact.
- [ ] **Continuous control via SB3 — PPO**: Swimmer, Walker2d, Hopper,
      HalfCheetah + manipulators (Reacher, Pusher). Train stock SB3 PPO, log
      V(s_t) along eval rollouts with a callback (mirror
      `GreedyEvalCallback` in `src/agents/sb3_fhr.py`), Hankel/AR analysis
      offline. Note: the parked PPO branch
      (`origin/claude/ppo-features-branch-pr-qwunww`) has a home-stack PPO but
      only classical configs — for *this* track SB3 is faster and needs no
      merge.
- [ ] **Continuous control via SB3 — SAC**: same envs; log Q(s_t, a_t) of the
      trained critics along on-policy rollouts (this is the "strap the Hankel
      analysis onto SAC" item). SAC also gives an off-policy continuous
      counterpoint to PPO.
- [ ] **Robustness knobs** for every env above: trained vs random policy,
      eps/sticky-action noise injection, 5 seeds — the claim is "low rank is a
      property of the (env, good policy) pair, robust to how you got there".

### Figures
- [ ] **F1.1 Spectra wall** — grid of normalised Hankel singular-value spectra
      (log y), one panel per env, spanning classical → Atari-100k → Atari →
      MuJoCo/PPO → SAC; effective-rank annotated per panel. The one-figure
      motivation.
- [ ] **F1.2 Rank vs training progress** — effective rank of the on-trajectory
      Hankel matrix over training for 3–4 representative envs: does low rank
      *emerge* with competence or is it there from the start?
- [ ] **F1.3 AR-order elbow** — held-out one-step and rolling-horizon error vs
      order r (probe already computes both), across envs; the elbow at r≈2
      justifies `fhr_order: 2` and the theory roots {1, 1/γ}.
- [ ] **F1.4 Robustness strip** — rank distributions across seeds/noise levels
      per env (box plots): tightness = robustness.

---

## Track 2 — FHR beyond DQN: PPO (GAE) and SAC variants

Bootstrapping the Hankel/AR regulariser into actor-critic learning. On-policy
is structurally *easier* for FHR: rollout buffers are already ordered
trajectories — no episodic-replay surgery needed.

### Variants to implement (try both, keep what works)
- [ ] **V-variant (PPO-FHR-V)**: FHR residual penalty on the value baseline
      V_φ over rollout segments, added to the PPO value loss. GAE itself is
      untouched; the penalty regularises the value fit that GAE consumes.
      Cheapest, most defensible.
- [ ] **Q-variant (PPO-FHR-Q)**: auxiliary Q- (or advantage-) head trained
      with TD + FHR alongside the GAE baseline. Two sub-modes:
      (a) pure auxiliary — shapes the shared trunk only;
      (b) blended advantages — mix GAE with Q-derived advantages
      (β-weighted), which is where the regularised Q can actually change the
      policy gradient.
- [ ] **SAC-FHR**: penalty on both critics over sampled episodic segments.
      Port `FHREpisodicReplayBuffer` (already solved for SB3 DQN in
      `src/agents/sb3_fhr.py`) to SAC's buffer; the recurrence head reuses
      `FHRRecurrenceHead` as-is.
- [ ] (stretch) **λ-return variant**: FHR directly on the empirical λ-return
      sequence rather than the network outputs — closest to the theory,
      probably noisiest.

### Decisions / notes
- Build on **SB3** (PPO + SAC) for breadth and trusted baselines — same
  pattern as the DQN comparison stack (λ=0 must stay bit-exact with stock
  SB3). Revive the home PPO branch only if we need deep diagnostics.
- Envs, in order: Pendulum/CartPole (debug) → Swimmer, Walker2d, Hopper →
  Reacher, Pusher (manipulators) → HalfCheetah/Ant if budget allows. This is
  also what "more environments to test" buys us for Track 3.

### Figures
- [ ] **F2.1** Learning curves per env: stock PPO vs PPO-FHR-V vs PPO-FHR-Q
      (seed bands); same for SAC.
- [ ] **F2.2** Value-function quality: explained variance / TD-error of the
      baseline over training with and without FHR (the mechanism figure —
      better baseline ⇒ lower-variance advantages).
- [ ] **F2.3** λ (fhr_weight) ablation heat strip per env.

---

## Track 3 — Claim 1: smaller effective optimisation space ⇒ better/faster convergence at equal samples

The discipline that makes the claim honest: identical sample budgets,
identical gradient-step budgets, identical everything except λ. Report
distributional aggregates, not means.

### Experiments
- [ ] **Atari-100k suite**: BBF-recipe baseline (`fhr_weight: 0`) vs +FHR,
      paired seeds, as many suite games as compute allows (Krull first,
      **[have]** EfficientRainbow Krull: 3310 → 5160 seed-0). All 27 configs
      are already on the BBF recipe with per-recipe manifest families.
- [ ] **Checkpointed sample-efficiency**: eval at 25k/50k/75k/100k steps (add
      periodic full-game eval to the launcher) so we can show *faster*, not
      just *better at the end*.
- [ ] **Classical control**: episodes-to-solve distributions (rolling-window
      threshold crossing, already logged) on MountainCar (**[have]** FHR 3/3
      vs baseline 0/3), CartPole (**[have]** wash — report it), Acrobot
      (**[have]** hurts — report it), + LunarLander to grow the set.
- [ ] **Manipulators/locomotion**: the Track-2 SAC/PPO-FHR arms double as
      Claim-1 evidence on continuous control.

### Figures
- [ ] **F3.1** Aggregate Atari-100k: IQM HNS + optimality gap with bootstrap
      CIs (rliable-style), baseline vs FHR; published methods as context axis
      (numbers already in `src/analysis/atari100k.py`).
- [ ] **F3.2** Per-game scatter: baseline HNS (x) vs FHR HNS (y), diagonal =
      no effect; one marker per game. Instantly readable win/loss figure.
- [ ] **F3.3** Sample-efficiency curves: score vs env steps at the eval
      checkpoints, per game and aggregated.
- [ ] **F3.4** Classical time-to-solve: per-seed dots + box per (env, arm);
      MountainCar is the headline panel.

---

## Track 4 — Claim 2: FHR lets *smaller* networks learn the same policies

Structure guides optimisation ⇒ fewer neurons needed. **[have]** the first
data point: MountainCar SB3 [32,32] vs archived [256,256] pass (this week's
manifests: `sb3_runs_manifest*_large_network_256_256.json` vs the new ones).

### Experiments
- [ ] **Capacity sweep, classical (SB3)**: net_arch ∈ {[16,16], [32,32],
      [64,64], [128,128], [256,256]} × {λ=0, λ*} × 5 seeds on MountainCar and
      CartPole; measure final return AND steps-to-solve.
- [ ] **Capacity sweep, Atari**: `width_scale` ∈ {1, 2, 4} and/or
      `head_hidden` ∈ {512, 1024, 2048} on 2–3 games (Krull + one FHR-win +
      one FHR-neutral game), baseline vs FHR.
- [ ] **Policy similarity check** ("similar policies", not just similar
      scores): action-agreement rate and return correlation between the small
      FHR net and the large baseline net on shared eval states — cheap to add
      to the eval pass.

### Figures
- [ ] **F4.1** Performance vs parameter count, two curves (baseline, FHR) per
      env — the claim is the FHR curve sits up-and-left.
- [ ] **F4.2** Smallest-net-that-solves bar per method.
- [ ] **F4.3** Action-agreement matrix (small-FHR vs large-baseline) per env.

---

## Track 5 — Soft claim: stability & hyperparameter insensitivity

Frame carefully: evidence-based "we observe", not a theorem. Known
counter-evidence stays in (Acrobot; the CartPole 1e-3 ceiling collapse hit
both arms) — the claim survives as "when the env's value sequence is
genuinely low-rank, FHR widens the stable region".

### Experiments
- [ ] **LR sensitivity sweep**: final performance vs learning rate (half-log
      grid) for baseline vs FHR on MountainCar + CartPole (SB3 stack —
      **[have]** the tuning study already showed polyak+lr1e-3+FHR winning
      where baseline degrades; formalise into a sweep).
- [ ] **Collapse statistics**: fraction of seeds that late-collapse after
      first solving (detector: rolling mean drops X% below its own best for
      ≥N episodes) across all multi-seed runs — mine the existing manifests
      first, top up seeds where the estimate is coarse.
- [ ] **Seed-variance shrinkage**: IQR of the learning curves at matched
      checkpoints, FHR vs baseline, across every env we've run — pure mining
      of **[have]** data.
- [ ] **Second knob** (target-update cadence or eps schedule) on one env, to
      show it isn't lr-specific.

### Figures
- [ ] **F5.1** Sensitivity curves: final score vs lr, two colours, per env;
      shaded stable region.
- [ ] **F5.2** Collapse-rate bars per (env, arm).
- [ ] **F5.3** Per-seed spaghetti overlays at the headline configs (the
      honest figure — every seed visible).

---

## Track 6 — Mechanism: seeing what the regulariser does to optimisation

The "why does this work" chapter: analyse weights/features in real time and
visualise the landscape. Best done on a designed toy env + one classical env
+ one Atari game.

### Experiments / instrumentation
- [ ] **Toy diagnostic env**: a small chain/ring MDP with *known* value AR
      order (rewards chosen so Q along the trajectory is exactly rank-2), tiny
      2-layer net — small enough to plot the actual loss surface and gradient
      field with and without the penalty. This is where "reduces the
      optimisation space" becomes a literal picture.
- [ ] **Learned-recurrence trajectory**: log the c/d coefficients every
      episode (already in diagnostics) and plot the roots of the learned AR
      polynomial over training vs the theory roots {1, 1/γ}. Real-time view:
      add a roots panel to the rank viewer.
- [ ] **Weight/feature spectra over training**: singular values of each layer
      (esp. the last linear) and of the feature matrix Φ over a probe batch,
      logged at the analysis cadence. Question: does FHR *raise* useful
      feature rank while lowering on-trajectory Hankel rank? Connect to the
      feature-rank-collapse literature (implicit under-parameterisation,
      capacity loss) — FHR as *targeted* structure, not collapse.
- [ ] **Loss-landscape slices** (filter-normalised 2D random directions, Li et
      al. 2018) around matched checkpoints, baseline vs FHR: is the FHR
      basin wider/smoother? Overlay both losses (TD alone vs TD+penalty) on
      the same slice.
- [ ] **Optimisation-path PCA**: project the weight trajectory onto its top-2
      PCs; plot baseline vs FHR paths over the TD-loss contours of the slice.
- [ ] **Mode connectivity**: linear interpolation barriers baseline↔FHR
      solutions (and small↔large nets from Track 4) — do FHR solutions live
      in the same basin, found faster?
- [ ] **Temporal feature structure**: cos-similarity of φ(s_t), φ(s_{t+k})
      along trajectories and AR-residual *in feature space* — FHR should make
      features temporally predictable, which is the claimed inductive bias.

### Figures
- [ ] **F6.1** Toy-env loss surface + gradient field, side by side (λ=0 vs
      λ>0), optimiser path overlaid.
- [ ] **F6.2** Learned AR roots converging to {1, 1/γ} over training (per env
      — beautiful and unique to this method).
- [ ] **F6.3** Layer-spectra ridgeline over training, baseline vs FHR.
- [ ] **F6.4** Landscape slice contours + optimisation paths.
- [ ] **F6.5** Temporal feature-similarity curves.

---

## Cross-cutting infrastructure
- [ ] Periodic full-game eval checkpoints in the Atari-100k launcher (Track 3).
- [ ] SB3 rollout/eval callback that dumps V/Q trajectories in the same npz
      format the analysis stack reads (Tracks 1, 2).
- [ ] rliable (or hand-rolled bootstrap IQM) helper in `src/analysis/`.
- [ ] One matplotlib style for all paper figures + a `figures/` manifest
      mapping figure id → generating notebook → data manifests, so every
      figure is reproducible from committed manifests.
- [ ] Weight/feature-spectrum logging hooks in the agents (Track 6).

## Suggested order of attack
1. **Track 1** figure pass on existing data (F1.1–F1.3) — no training, pure
   consolidation, and it's the motivation figure.
2. **Track 3** Atari-100k: keep BBF±FHR seeds running (Krull → 4–6 games),
   add checkpoint evals early so curves accumulate.
3. **Track 4** classical capacity sweep (cheap, SB3, overnight) — finishes
   Claim 2's first real figure.
4. **Track 2** PPO-FHR-V on Pendulum/Swimmer (the smallest new
   implementation), then SAC critic logging for Track 1d en route.
5. **Track 5** mining pass on existing manifests; top-up sweeps after.
6. **Track 6** toy env + roots figure (F6.1/F6.2) whenever a GPU is busy —
   it's CPU-friendly.
