# The Low-Rank RL Ideas, Explained From Scratch

*A plain-language guide to what you've discovered so far, the six new algorithm ideas, and the experiments about to run.*

---

## Part 0 — The concepts you need (5 minutes)

**Reinforcement learning in one paragraph.** An agent lives in an environment (CartPole, LunarLander…). At each step it sees a *state*, picks an *action*, gets a *reward*. Its goal: maximize total reward per episode. Everything below is about two families of methods you've used: **DQN** (learn a table/network of "how good is each action here" and act greedily on it) and **PPO** (learn a *policy* — a probability of each action — directly, nudged in the direction that made returns higher).

**The value function V(s).** "If I'm in state s and play on from here, how much total reward do I expect?" PPO learns this as a helper network called the **critic**. The policy network is the **actor**. The critic never picks actions — it only *scores* states so the actor knows whether an action turned out better or worse than expected (that difference is the **advantage**).

**The key empirical discovery of your thesis.** Write down the critic's value estimates along one episode: v₁, v₂, v₃, … You found that this sequence is *extremely predictable*: each value is almost exactly a fixed weighted sum of the previous two:

> v_t ≈ a·v_{t−1} + b·v_{t−2}   (an "AR(2) recurrence" — two numbers a, b describe the whole curve)

**"Low Hankel rank" is just the fancy name for this.** Take the value sequence, stack overlapping windows of it into a matrix (the *Hankel matrix*). If the sequence obeys an order-2 recurrence, that matrix has rank ≤ 2 — nearly all of its singular values are zero. So "low Hankel rank" = "the value signal follows a simple two-term echo rule". Your measurement: an AR(2) fit forecasts held-out value curves better than the critic itself predicts its own future values (Acrobot: 10% error at 64 steps vs 27% for a naive forecast).

**Why anyone would care.** If value curves *must* look like simple recurrences, you can use that as a prior: penalize the critic for producing wiggly, recurrence-breaking curves (that's a *regularizer*), or denoise its training targets toward the recurrence, or build networks that can only produce such curves. A good prior = learning from fewer samples = **sample efficiency**.

---

## Part 1 — What you already proved (and disproved)

| Campaign | What was tried | Outcome |
|---|---|---|
| **HR-DQN** | Penalize the Q-network when its value windows have high Hankel rank | ✅ **Worked**: +27% area-under-learning-curve, p=0.0015 |
| **Value-recurrence study** | Measure how predictable value sequences are | ✅ AR(2) beats the critic's own bootstrap; HR-DQN values 5× more self-predictable |
| **PPO campaign** (4 mechanisms, ~410 runs, everything pre-registered) | Same penalty on PPO's critic; AR-filtered advantages; AR bootstrap at rollout cuts; Cadzow-denoised critic targets | ❌ **All null or harmful** on replication |

The post-mortem gave three hard-won lessons that shape everything below:

1. **Put the structure where it drives behavior.** In DQN the value function *is* the policy (argmax over it), so shaping values shapes behavior — it worked. In PPO the critic is a side-channel (a variance reducer for the actor); shaping it barely touches what the agent *does*.
2. **On these environments, luck between seeds comes from exploration**, not critic noise. A mechanism that never influences *which states get visited* can't move the outcome distribution.
3. **The prior breaks on spiky signals.** LunarLander returns have crash/landing spikes — genuinely high-rank events. Rank-2 smoothing erased them and destroyed learning. The prior must be *robust* or *adaptive*, not blindly applied.

The six ideas below are six different ways to respect those lessons.

---

## Part 2 — The six ideas, in plain language

### Idea 1 — SSM critic: build the recurrence into the network's wiring (`ssm_critic`)

Instead of *punishing* the critic when its outputs break the recurrence (the approach that failed), make a critic that is **physically only capable of producing recurrence-shaped outputs**. The critic becomes a tiny *linear recurrent unit*: it carries a small hidden memory h (a handful of numbers), and at each step updates it by a fixed linear rule, h_t = A·h_{t−1} + B·features(s_t), and reads the value off it, v_t = C·h_t. Any output of such a machine automatically satisfies a low-order recurrence — the prior holds *by construction*, and spikes can still enter through the input features rather than being smeared away.

This is the same mathematical core as the "structured state-space models" (S4/Mamba) that recently revolutionized long-sequence modeling. Analogy: rather than fining a driver every time they swerve (penalty), you put them on rails (architecture).

**Test:** does a rail-critic PPO learn faster than vanilla PPO?

### Idea 2 — Latent-AR representations: teach the *features* to evolve simply (`latent_ar`)

Your penalty acted on the critic's final scalar output — the very end of the pipeline. This idea moves the prior upstream: the network's internal *representation* of the state (a vector z of ~64 numbers) is trained with an auxiliary objective: "z_{t+1} should be predictable as a fixed linear function of z_t and z_{t−1}." The actor and critic **share** this representation — so unlike in the failed campaign, the structure now reaches the part of the network that picks actions. This is the "self-predictive representations" idea from the SPR/Koopman literature, specialized to your order-2 discovery.

Analogy: instead of demanding the essay's final sentence be elegant, teach the writer a simple grammar — everything downstream inherits it.

**Test:** does the shared-representation auxiliary loss speed up learning?

### Idea 3 — Recurrence-violation as an exploration compass (`ar_explore`)

Your own conclusion was that seed luck = exploration luck. So point the structure at exploration: fit the two-number AR(2) rule to each rollout's value curve (microseconds), then measure, at each step, how badly the actual value *broke* the rule. Big violation = "something surprising/unmodeled happened here" = a place worth visiting more. Add a small bonus reward proportional to the violation. This is the classic "curiosity" recipe (RND/ICM), but with your recurrence as the novelty detector — nearly free to compute, no extra networks.

Analogy: a metal detector that beeps where the terrain stops matching the map — and you deliberately dig where it beeps.

**Test:** does the violation bonus improve exploration-limited environments (Acrobot especially)?

### Idea 4 — Robust denoising: smooth the trend, keep the spikes (`robust_hd`)

HD-PPO failed because it forced *everything* — including real crash/landing events — onto the smooth low-rank curve. The fix comes from "Robust PCA": model the return sequence as **smooth low-rank part + sparse spikes**. Iterate: pull out the few big outliers first (the spikes), denoise only what remains toward rank-2, then add the spikes back untouched. The critic's targets get cleaner *without* losing the events that matter.

Analogy: noise-cancelling headphones that suppress hiss but are explicitly built to let the fire alarm through.

**Test:** does spike-preserving denoising help where plain denoising was null (CartPole) — and, later, not hurt where it was harmful (LunarLander)?

### Idea 5 — Phasic training: give the value structure a road into the actor (`ppg_lite`)

Standard PPO trains actor and critic as separate networks — which is exactly why your value-side tricks never reached the actor. Phasic Policy Gradient (PPG) alternates two phases: a *policy phase* (normal PPO) and an *auxiliary phase* where value-learning signal is distilled **into the actor's own network** (with a constraint stopping the policy from drifting). If the value function carries your structure, the aux phase is the pipe that carries it into behavior.

Analogy: the coach (critic) usually just scores the player's games; in PPG, between matches, the coach gets dedicated training sessions with the player.

**Test:** does a lightweight PPG (shared trunk, periodic aux phase) beat vanilla PPO here — establishing the host into which structure priors could later be added?

### Idea 6 — The recurrence as *transferable knowledge* (analysis probe)

The AR(2) rule is two numbers. Are they a property of the *environment* (same for every training run) rather than the run? If yes, they're the cheapest transferable object imaginable: fit them once on a source task, hand them to a fresh agent on a related task (e.g., to power Idea 3's compass from step one). This ties the campaign back to the thesis title — *transfer* learning. This round we run it as an analysis: measure coefficient stability across seeds and environments, and test source-fitted coefficients driving `ar_explore` on a different environment.

---

## Part 3 — How the experiment will be judged (no fooling ourselves)

The protocol copies the discipline that made your previous negative results trustworthy:

1. **Exploration round (cheap):** every variant vs. vanilla PPO baseline on CartPole-v1 and Acrobot-v1, ~5 seeds each, short budgets, identical seeds across arms (paired comparison). Purpose: find signal and sane hyperparameters — *not* to make claims.
2. **Confirmation round (the only round that counts):** the most promising variants re-run on **fresh seeds**, N ≥ 10 per arm. Primary metric declared before launch: **AUC of the episode-return learning curve** (this *is* sample efficiency — how much reward accumulated per episode of experience), tested with a one-sided permutation test vs. baseline, p < 0.05. Secondary: final-quarter mean return. Anything that only wins in round 1 gets reported as "did not replicate," exactly like rounds A–G.
3. **Deliverables:** learning curves (mean ± 95% CI over seeds), AUC bar charts with p-values, and an honest write-up — pushed to a `fable/test` branch only, nothing touches `main` or PR #29.

**A caution worth setting now:** your own campaign history says most small-N wins die on confirmation. If some ideas come back null, that is the method working. The design above maximizes the chance that whatever survives is real.
