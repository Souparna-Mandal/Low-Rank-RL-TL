# HR-DQN: Hankel-Rank-Regularised DQN

A modification of the classical DQN loss that adds a differentiable penalty on the
**Hankel rank of predicted $Q_{\pi}(s,a)$ or $V_{\pi}(s)$ along replayed sub-trajectories**. 

Changes are: 

1. [`src/agents/hankel_dqn_agent.py`](../src/agents/hankel_dqn_agent.py) (agent + episodic
replay buffer) 
2. [`src/agents/hankel_regulariser.py`](../src/agents/hankel_regulariser.py)
(the penalty)

3. benchmarked in `experiments/dqn_cartpole/exp1_hankel.ipynb` and
`experiments/dqn_acrobot_2_revised/exp1_hankel.ipynb`.

## Motivation 

The analysis runs in this repo show that Hankel matrices built from value/Q signals along rollouts of the *learned  policy* are consistently low-rank —
CartPole ≈ rank 1 — and the same low rank shows up on **sub-trajectories**, so the property is local and measurable on short replay windows. HR-DQN bets on the inverse direction, since the property holds upon convergence and during training we believe that *enforcing it during
training* biases the search toward the region of function space where good solutions were observed to live. 

What the penalty means is plain linear algebra: Hankel rank ≤ r
⟺ the sequence approximately obeys an order-r linear recurrence, i.e. the value
signal is predictable from its own recent past. The claim is *biasing*, not a hard
restriction of the function class.

## The loss

Replay stores whole episodes. Two samplers feed training: i.i.d. transitions for the
TD loss (at λ = 0 the agent reproduces `QAgent` exactly), and episode-contiguous
windows $w = (s_t, a_t)_{t=\tau}^{\tau+T-1}$ for the penalty.

For a window $w$, the predicted value sequence and its Hankel lift are

$$\mathbf{q}(w;\theta) = \big(Q_\theta(s_\tau,a_\tau), \dots, Q_\theta(s_{\tau+T-1},a_{\tau+T-1})\big) \in \mathbb{R}^T,
\qquad
\mathcal{H}(\mathbf{q})_{ij} = q_{i+j-1} \in \mathbb{R}^{(T-L+1)\times L},\; L=\lceil T/2\rceil .$$

The ideal objective penalises $\operatorname{rank}\mathcal{H}(\mathbf{q})$, which is
untrainable (integer-valued, zero gradient a.e.). The surrogate chain:

- The nuclear norm $\|\mathcal{H}\|_* = \sum_i \sigma_i$ is the standard convex
  relaxation of rank, but shrinks **all** singular values — i.e. shrinks the scale of
  $Q$ itself, fighting the TD loss. Rejected as-is.
- We want rank ≤ r (measured: r = 2 on CartPole), not rank 0, so we penalise only the
  spectrum's **tail beyond r** (truncated nuclear norm): $\sum_{i>r}\sigma_i$, which is
  zero **iff** $\operatorname{rank}\mathcal{H} \le r$ and leaves the top-r subspace
  (the signal) untouched.
- Normalise for scale-invariance with a stop-gradient denominator (else the optimiser
  cheats by inflating $\sigma_{1..r}$):

$$\mathcal{R}_r(\mathbf{q}) = \frac{\sum_{i>r}\sigma_i(\mathcal{H}(\mathbf{q}))}{\operatorname{sg}\big(\sum_i \sigma_i(\mathcal{H}(\mathbf{q}))\big)},
\qquad \mathcal{R}_r = 0 \iff \operatorname{rank}\mathcal{H}(\mathbf{q}) \le r .$$

Full training loss, with the double-DQN target
$y = r + \gamma(1-d)\,Q_{\bar\theta}(s', \arg\max_{a'} Q_\theta(s',a'))$:

$$\mathcal{L}(\theta) =
\underbrace{\mathbb{E}_{(s,a,r,s',d)}\big[\mathrm{Huber}(Q_\theta(s,a), y)\big]}_{\text{classical DQN TD loss}}
\;+\; \lambda_k\,
\underbrace{\mathbb{E}_{w}\big[g(w)\cdot \mathcal{R}_r(\mathbf{q}(w;\theta))\big]}_{\text{low-rank Hankel regulariser}}$$

where $g(w) = \mathbb{1}\{\text{relative tail energy} \le \rho\}$ is a no-grad gate
that skips off-manifold windows, and $\lambda_k$ ramps linearly from 0 after a warm-up.

## Pseudocode (LaTeX, algorithmicx)

```latex
\begin{algorithm}[t]
\caption{Hankel-Rank-Regularized DQN (HR-DQN)}
\label{alg:hrdqn}
\begin{algorithmic}[1]
\Require target rank $r$, penalty weight $\lambda$, window length $T$, windows per step $B_w$,
         TD batch size $B$, gate threshold $\rho$, warm-up steps $K_0$, discount $\gamma$
\State Initialise $Q_\theta$, target $Q_{\bar\theta} \gets Q_\theta$, episodic replay $\mathcal{D}$
\For{each environment step}
    \State Act $\varepsilon$-greedily, append transition to current episode; on episode end, store episode in $\mathcal{D}$
    \State \textbf{TD loss:} sample i.i.d. $\{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^{B} \sim \mathcal{D}$
    \State $y_i \gets r_i + \gamma\,(1-d_i)\, Q_{\bar\theta}\big(s'_i,\ \arg\max_{a'} Q_\theta(s'_i,a')\big)$
    \State $\mathcal{L}_{\mathrm{TD}} \gets \tfrac{1}{B}\sum_i \mathrm{Huber}\big(Q_\theta(s_i,a_i),\, y_i\big)$
    \State \textbf{Hankel penalty:} sample $B_w$ contiguous windows $w_b=(s_t,a_t)_{t=\tau_b}^{\tau_b+T-1}$,
           each within a single episode, excluding terminal-crossing windows
    \For{each window $w_b$}
        \State $\mathbf{q}_b \gets \big(Q_\theta(s_t,a_t)\big)_{t=\tau_b}^{\tau_b+T-1} \in \mathbb{R}^{T}$
        \State $H_b \gets \mathrm{Hankel}(\mathbf{q}_b) \in \mathbb{R}^{(T-L+1)\times L}$
               \Comment{$H_{ij} = q_{i+j-1}$, $L=\lceil T/2\rceil$}
        \State $\sigma_1 \ge \dots \ge \sigma_{\min(T-L+1,L)} \gets \mathrm{svdvals}(H_b)$
        \State $\varrho_b \gets \sum_{i>r}\sigma_i \,/\, \sum_i \sigma_i$
               \Comment{relative tail energy}
        \State $g_b \gets \mathbb{1}\{\varrho_b \le \rho\}$
               \Comment{gate: skip off-manifold windows (no-grad)}
        \State $R_b \gets \sum_{i>r}\sigma_i \,/\, \mathrm{sg}\big(\sum_i \sigma_i\big)$
               \Comment{$\mathrm{sg}$ = stop-gradient on the denominator}
    \EndFor
    \State $\mathcal{R}_{\mathrm{Hankel}} \gets \big(\sum_b g_b R_b\big) / \max\big(\sum_b g_b,\,1\big)$
    \State $\lambda_k \gets \lambda \cdot \min\big(1,\ \max(0,\ k - K_0)/K_{\mathrm{ramp}}\big)$
           \Comment{warm-up ramp at grad step $k$}
    \State $\mathcal{L} \gets \mathcal{L}_{\mathrm{TD}} + \lambda_k\, \mathcal{R}_{\mathrm{Hankel}}$
    \State $\theta \gets \theta - \eta\,\mathrm{clip}\big(\nabla_\theta \mathcal{L}\big)$;
           \quad $\bar\theta \gets (1-\tau_{\mathrm{polyak}})\bar\theta + \tau_{\mathrm{polyak}}\theta$
\EndFor
\end{algorithmic}
\end{algorithm}
```

## Why this design (assessment + caveats)

1. **The empirical argument is coherent.** The property was observed at the solution,
   holding even on sub-trajectories; penalising its violation shrinks the effective
   search space without excluding the solutions actually found.
2. **Off-policy mismatch.** The observation is for the learned policy's own greedy
   closed-loop rollouts; replayed windows come from older weights acting ε-greedily,
   so the penalty evaluates *current* Q along *old, occasionally ε-kicked* state paths
   (with ε = 0.05, T = 16, only ~44% of windows are kick-free). Tolerable — policies
   drift slowly, ε has decayed by the time the ramped λ activates, and sub-trajectory
   locality is what licenses short replay windows as measurement sites. The gate
   doubles as an off-policy filter: kicked/stale windows show high tail energy and are
   skipped. If `gate_frac` stays persistently high, escalate to recency-biased window
   sampling: set `window_half_life` (in episodes) so penalty windows are drawn with
   weight 0.5^(age/half_life) — the TD sampler stays buffer-wide either way.
3. **Do not penalise toward rank 0.** Hence tail-beyond-r, relative normalisation,
   detached denominator; r should come from the *measured* rank in the analysis
   sweeps, not be assumed.
4. **The observation is about the converged policy.** Mid-training Q along a window is
   mostly noise, and the penalty gradient there is arbitrary — hence the λ warm-up
   ramp and the per-window gate (both benchmarked as variants). Also monitor feature
   srank (Kumar et al., arXiv 2010.14498: *feature-matrix* rank collapse hurts — a
   different object than temporal Hankel rank, but worth verifying the penalty does
   not induce it).

**Literature positioning (checked July 2026):** nearest lanes are Hankel nuclear-norm
regularisation for system identification (Sun & Oymak, arXiv 2203.16673), Hankel
singular-value regularisation for SSM compression (arXiv 2510.22951), and *spatial*
low-rank Q-matrix penalties (arXiv 2111.10103; SV-RL, arXiv 1909.12255). Penalising
the *temporal* Hankel rank of value sequences during value-based RL training appears
unoccupied.

## Alternatives considered

- **Recurrence-residual penalty** (future): fit order-r AR coefficients per window by
  ridge lstsq on the detached sequence, penalise the live residual. No SVD; enforces
  exactly the order-r recurrence; the fitted roots are the value signal's discrete-time
  modes. Con: coefficients lag the network within a step.
- **Detached-projection residual** (future): $\|(I-P_r)\mathcal{H}(\mathbf{q})\|_F^2$
  with $P_r$ from a no-grad SVD (Eckart–Young). Avoids SVD backward; projection frozen
  within the step.
- **Target-side Hankel denoising**: project bootstrap target sequences onto the
  nearest rank-r Hankel matrix and regress on them. No gradient through SVD, but a
  *hard* constraint that biases when sequences are off-manifold, and it shapes targets
  rather than the function class.
- **Log-det surrogate** (rejected): δ-sensitivity, gradient blow-up as σ → 0,
  penalises all directions.
- **Spectral-entropy / effective-rank penalty** (rejected for now): scale-invariant
  but pushes spectra toward degeneracy — the SVD-backward worst case.
- **Spatial low-rank on the batch Q-matrix** (contrast only): for |A| = 2 the (N×2)
  Q-matrix is shape-capped at rank 2 — precisely why the temporal axis is the novel one.

## Implementation notes

- `EpisodicReplayBuffer`: episodes are the storage unit; `sample_transitions` is
  uniform over stored transitions (including the in-progress episode), matching
  `ReplayBuffer` in distribution; `sample_windows` returns fixed-length single-episode
  windows and never crosses a reset. Capacity counted in transitions, FIFO by episode.
- `HankelDQNAgent(QAgent)` config keys (under `agent:` — the constructor-kwarg
  convention): `hankel_weight, hankel_order, window_len, n_windows, gate_threshold,
  warmup_grad_steps, ramp_grad_steps, penalize_terminal_windows, td_source,
  hankel_log`. `hankel_log: true` computes the penalty on the signed log
  sign(v)·log1p(|v|) of each window's value sequence (values can be negative/zero,
  so a plain log is undefined) instead of the raw values.
  `td_source: windows` is the ablation where TD pairs come from inside the penalty
  windows (correlated batches; terminal anchors preserved via each window's
  post-window state).
- Numerics: SVD runs in float64 (via CPU on MPS — float64 is unsupported there, in
  both forward and the backward replay); windows already at rank ≤ r or above the gate
  get zero cotangents (svdvals' backward $U\,\mathrm{diag}(g)\,V^\top$ stays finite for
  degenerate spectra); non-finite losses/grad-norms skip the step and increment
  `nan_skips`.
- Diagnostics per `train()` (logged to `train_diagnostics.csv` via the run logger, or
  captured by the benchmark): `td_loss, lambda_eff, penalty_raw, penalty_weighted,
  gate_frac` (above ρ), `converged_frac` (tail ≈ 0), `batch_eff_rank` (0.999 energy-rank
  of the penalty windows), `rel_tail, nan_skips`.

## Benchmark

`experiments/{dqn_cartpole,dqn_acrobot_2_revised}/exp1_hankel.ipynb`: variants
`baseline` (λ=0 ≡ classical DQN), `tail_lo` (λ=1e-3), `tail_hi` (λ=1e-2), `gated`
(λ=1e-2, ρ=0.25, warm-up+ramp), `windows_lo` (single-sampler ablation) × seeds 0–3,
cached under `results_hankel/`. Per run: episode rewards, 20-episode greedy final
eval, the diagnostics trajectories, and the final on-policy Hankel-Q effective rank.
Tests: `python tests/test_hankel_dqn.py`.

### Results (July 2026, 40 runs, 0 failures, 0 nan-skips)

| variant | CartPole eval20 | Acrobot eval20 |
|---|---|---|
| baseline (λ=0) | 490.0 ± 14.2 | −91.2 ± 5.2 |
| tail_lo (λ=1e-3) | **500.0 ± 0.0** (all 4 seeds) | −87.5 ± 10.0 |
| tail_hi (λ=1e-2) | 498.6 ± 2.5 | **−82.2 ± 5.0** |
| gated (λ=1e-2, ρ=0.25) | 494.5 ± 8.0 | **−82.3 ± 4.7** |
| windows_lo (single-sampler) | 11.8 ± 4.4 | −167.6 ± 72.3 |

The penalty never hurt and the best variant beat baseline on both envs; the
effective λ is env-dependent (light for CartPole, strong for Acrobot). Forensics:
replay windows sit near the low-rank manifold from early training (relative tail
≈ 1.5–2% → ≈ 0 under the penalty), `gate_frac` ≈ 0 throughout (no off-policy
contamination; `window_half_life` unused), and final on-policy energy-rank
saturates at 1 for every variant — but that was measured at the old 0.90 energy
threshold, which reports rank 1 for any spectrum whose leading mode holds 90% of
the energy (an order-5 recurrence included), so it is a property of the cutoff as
much as of the runs. The diagnostic now uses 0.999; re-measure before citing it. The `windows_lo` collapse shows the separate
i.i.d. TD sampler is load-bearing: window-TD batches are both correlated (~8
independent segments/step) and biased (episodes shorter than the window never
enter the TD loss, so failures are never learned from).
