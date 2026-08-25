"""BBFAgent — the BBF training recipe (Schwarzer et al. 2023, arXiv:2305.19452)
on this repo's EfficientRainbow stack, with the FHR penalty available as the
auxiliary loss (``fhr_weight: 0`` is the pure-recipe baseline arm).

Everything below is ported from the official ``bbf/configs/BBF.gin`` +
``bbf/agents/spr_agent.py`` (google-research, Apache-2.0), the shipped RR=2
configuration:

  * Impala-CNN ResNet encoder at width 4 with per-sample feature
    renormalisation (``atari_networks.ImpalaCNNEncoder``), dense head 2048.
  * AdamW lr 1e-4, eps 1.5e-4, weight decay 0.1 masked to ndim>=2 parameters
    (biases exempt — BBF's ``decay_bias=False``).
  * EMA target network, tau 0.005 per GRADIENT step (the training loop's
    ``update_target_network`` tick is a no-op here), and target-network action
    selection (SR-SPR's ``target_action_selection``).
  * n-step annealed 10 -> 3 and gamma annealed 0.97 -> 0.997, both
    log-linearly over the first ``anneal_grad_steps`` (10k) gradient steps
    FOLLOWING EACH RESET (the schedules restart every reset).
  * Shrink-and-perturb every ``reset_interval_grad_steps`` (40k) gradient
    steps: the IQN head (everything outside ``policy_net.encoder``) is fully
    re-initialised, the encoder moves ``perturb_factor`` (50%) toward a fresh
    random init; the target network gets the identical treatment from the
    same fresh draw; Adam moments are zeroed for the reset head and kept for
    the shrink-perturbed encoder. The FHR c/d coefficients are agent-level
    parameters and deliberately survive resets.
  * epsilon-greedy exploration (no noisy nets, per BBF), decayed LINEARLY
    from eps_start to eps_min over ``eps_decay_steps`` (2001) post-warm-up
    env steps — overriding the exponential ``decay_rate`` scheme, whose
    config value is ignored.
  * No gradient clipping in BBF: set ``grad_clip_norm: .inf`` in the config
    (the clip call then only computes the norm for the NaN guard).

Deliberate deviations from published BBF, per this repo's conventions:
  * IQN-8 dueling head instead of C51-51 (thin-encoder contract; keeps the
    quantile stack shared with EfficientRainbowAgent).
  * Uniform episodic replay instead of prioritized (PER over the episodic
    buffer is stubbed repo-wide; the FHR penalty needs the episodic handles).
  * No SPR self-predictive auxiliary loss — the point of the comparison: the
    FHR recurrence penalty is the auxiliary under test on top of BBF's TD
    loss. ``replay_buffer_capacity`` 100k == BBF's 200k in effect (a 100k-step
    run never stores more than 100k transitions).
"""
import torch
import torch.optim as optim

from .efficient_rainbow_agent import EfficientRainbowAgent


class BBFAgent(EfficientRainbowAgent):
    """See the module docstring. Extra kwargs over EfficientRainbowAgent (all
    mappable to config ``agent:`` keys; defaults = the official BBF.gin):

        target_ema_tau: per-gradient-step EMA coefficient of the target net.
        target_action_selection: act (train AND eval) with the EMA target
            network instead of the online network.
        gamma_start: annealed discount start; ``discount_factor`` is the END
            value (0.997 in BBF).
        n_step_final: annealed n-step end; ``n_step`` is the START (10).
        anneal_grad_steps: length of each post-reset annealing phase.
        reset_interval_grad_steps: shrink-and-perturb cadence (0 disables).
        shrink_factor / perturb_factor: encoder interpolation weights
            theta <- shrink*theta + perturb*theta_random.
        eps_decay_steps: linear epsilon decay length in env steps (the
            inherited exponential decay_rate is ignored).
    """

    def __init__(self, *, target_ema_tau: float = 0.005,
                 target_action_selection: bool = True,
                 gamma_start: float = 0.97, n_step_final: int = 3,
                 anneal_grad_steps: int = 10000,
                 reset_interval_grad_steps: int = 40000,
                 no_resets_after_grad_steps: int | None = None,
                 shrink_factor: float = 0.5, perturb_factor: float = 0.5,
                 eps_decay_steps: int = 2001,
                 torch_compile: bool = False,
                 **era_kwargs):
        if not 0.0 < target_ema_tau <= 1.0:
            raise ValueError(f"target_ema_tau must be in (0, 1], got {target_ema_tau}")
        if n_step_final < 1:
            raise ValueError(f"n_step_final must be >= 1, got {n_step_final}")
        if not 0.0 < gamma_start < 1.0:
            raise ValueError(f"gamma_start must be in (0, 1), got {gamma_start}")
        if reset_interval_grad_steps < 0 or anneal_grad_steps < 0:
            raise ValueError("reset/anneal grad-step counts must be >= 0")
        if not (0.0 <= shrink_factor <= 1.0 and 0.0 <= perturb_factor <= 1.0):
            raise ValueError("shrink_factor/perturb_factor must be in [0, 1]")
        super().__init__(**era_kwargs)
        if n_step_final > self.n_step:
            raise ValueError(f"n_step_final {n_step_final} > n_step {self.n_step}")

        # BBF reference init (spr_networks.py): xavier_uniform kernels, zero
        # biases — applied to the initial nets AND to every fresh draw the
        # shrink-and-perturb resets interpolate toward (the perturbation scale
        # is part of the recipe).
        base_factory = self._net_factory
        self._net_factory = lambda: self._xavier_reinit(base_factory())
        self._xavier_reinit(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_ema_tau = target_ema_tau
        self.target_action_selection = target_action_selection
        self.gamma_start = gamma_start
        self.n_step_final = n_step_final
        self.anneal_grad_steps = anneal_grad_steps
        self.reset_interval_grad_steps = reset_interval_grad_steps
        self.no_resets_after_grad_steps = no_resets_after_grad_steps
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor
        self.eps_decay_steps = eps_decay_steps
        self._eps_start_value = self.epsilon
        self._last_reset = 0
        self.reset_count = 0

        # Rebuild the optimiser with BBF's bias-exempt weight decay: the
        # QAgent-built group is split into (ndim>=2, wd) + (ndim<2, wd 0);
        # the c/d group FHRDQNAgent appended stays LAST with its own lr and
        # wd 0 — the invariant the FHR tests pin.
        g0, cg = self.optimiser.param_groups[0], self.optimiser.param_groups[-1]
        decay = [p for p in self.policy_net.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.policy_net.parameters() if p.ndim < 2]
        on_cuda = next(self.policy_net.parameters()).is_cuda
        # fused=True: single-kernel AdamW — the elementwise multi-pass update
        # over ~90M params is memory-bandwidth-bound on unified-memory GPUs
        # (profiled at ~30% of a gradient step's CUDA time unfused).
        self.optimiser = optim.AdamW(
            [{"params": decay, "weight_decay": g0["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0},
             {"params": cg["params"], "lr": cg["lr"], "weight_decay": 0.0}],
            lr=g0["lr"], eps=g0["eps"], amsgrad=g0["amsgrad"], fused=on_cuda)

        # One fused lerp per EMA tick instead of ~50 sequential mul_/add_
        # pairs. The lists survive resets: _shrink_perturb_reset mutates these
        # SAME tensors in place, never rebinds them.
        self._ema_tgt = [t for _, t in sorted(self.target_net.state_dict().items())]
        self._ema_src = [s for _, s in sorted(self.policy_net.state_dict().items())]

        self.torch_compile = torch_compile
        if torch_compile and on_cuda:
            # quantiles is the single hot path (TD online/target forwards,
            # Double-Q selection, acting, FHR penalty all route through it);
            # dynamic=True because the non-terminal row count varies per batch.
            self.policy_net.quantiles = torch.compile(
                self.policy_net.quantiles, dynamic=True)
            self.target_net.quantiles = torch.compile(
                self.target_net.quantiles, dynamic=True)

    # ------------------------------------------------------------ schedules
    def _cycle_frac(self) -> float:
        """Fraction of the current annealing phase completed (gradient steps
        since the last reset / anneal_grad_steps, capped at 1)."""
        if self.anneal_grad_steps <= 0:
            return 1.0
        return min(1.0, (self._grad_steps - self._last_reset) / self.anneal_grad_steps)

    def _current_gamma(self) -> float:
        g0, g1 = self.gamma_start, self.loss.gamma
        if g0 == g1:
            return g1
        # log-linear in (1 - gamma): 0.03 -> 0.003 in BBF
        return 1.0 - (1.0 - g0) * ((1.0 - g1) / (1.0 - g0)) ** self._cycle_frac()

    def _current_n_step(self) -> int:
        n0, n1 = self.n_step, self.n_step_final
        if n0 == n1:
            return n0
        return max(n1, int(round(n0 * (n1 / n0) ** self._cycle_frac())))

    # ----------------------------------------------- exploration and acting
    def decay_epsilon(self):
        """BBF: linear decay over eps_decay_steps post-warm-up env steps
        (called once per such step by the training loop)."""
        if self.eps_decay_steps and self.eps_decay_steps > 0:
            step = (self._eps_start_value - self.eps_min) / self.eps_decay_steps
            self.epsilon = max(self.eps_min, self.epsilon - step)
        else:
            super().decay_epsilon()

    def act_greedy(self, state: torch.Tensor):
        net = self.target_net if self.target_action_selection else self.policy_net
        return net(state).argmax(dim=1).item()

    # ------------------------------------------------------- target network
    def update_target_network(self):
        """No-op: BBF's EMA target updates ride every gradient step inside
        _post_optim_step; the training loop's env-step tick must not add a
        second (TD_LR-weighted) update on top."""

    def _post_optim_step(self):
        with torch.no_grad():
            torch._foreach_lerp_(self._ema_tgt, self._ema_src,
                                 self.target_ema_tau)
        if (self.reset_interval_grad_steps > 0 and
                self._grad_steps - self._last_reset >= self.reset_interval_grad_steps
                and (self.no_resets_after_grad_steps is None
                     or self._grad_steps <= self.no_resets_after_grad_steps)):
            self._shrink_perturb_reset()

    # ------------------------------------------------------ periodic resets
    @staticmethod
    def _xavier_reinit(net):
        for m in net.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        return net

    def _shrink_perturb_reset(self):
        """BBF's reset: head fully re-initialised, encoder interpolated toward
        a fresh random init; the target network gets the identical treatment
        from its OWN independent fresh draw (reference behaviour); Adam
        moments zeroed for the reset head only. Restarts the n-step/gamma
        annealing cycle."""
        with torch.no_grad():
            for net in (self.policy_net, self.target_net):
                fresh_params = dict(
                    self._net_factory().to(self.device).named_parameters())
                for name, p in net.named_parameters():
                    fp = fresh_params[name]
                    if name.startswith("encoder."):
                        p.mul_(self.shrink_factor).add_(fp, alpha=self.perturb_factor)
                    else:
                        p.copy_(fp)
        head_ids = {id(p) for n, p in self.policy_net.named_parameters()
                    if not n.startswith("encoder.")}
        for p in [p for p in self.optimiser.state if id(p) in head_ids]:
            del self.optimiser.state[p]
        self._last_reset = self._grad_steps
        self.reset_count += 1

    # ------------------------------------------------------------- learning
    def _train_step(self):
        gamma_eff, n_eff = self._current_gamma(), self._current_n_step()
        diag = super()._train_step()
        diag["gamma_eff"] = gamma_eff
        diag["n_step_eff"] = float(n_eff)
        diag["resets"] = float(self.reset_count)
        return diag

    # ---------------------------------------------------------- persistence
    def save(self, path):
        super().save(path)
        payload = torch.load(path, map_location="cpu")
        payload.update({"bbf_grad_steps": self._grad_steps,
                        "bbf_last_reset": self._last_reset,
                        "bbf_reset_count": self.reset_count})
        torch.save(payload, path)

    def load(self, path):
        super().load(path)
        ckpt = torch.load(path, map_location=self.device)
        if "bbf_grad_steps" in ckpt:
            self._grad_steps = int(ckpt["bbf_grad_steps"])
            self._last_reset = int(ckpt["bbf_last_reset"])
            self.reset_count = int(ckpt["bbf_reset_count"])
