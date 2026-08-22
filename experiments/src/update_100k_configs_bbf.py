"""One-off migration: switch all 27 Atari-100k game configs to the BBFAgent
recipe (arXiv:2305.19452 — Impala-CNN x4 + EMA target + n-step/gamma
annealing + shrink-and-perturb resets + DrQ augmentation, official BBF.gin
RR=2 hyperparameters; IQN-8 head and uniform replay per repo conventions).

Exact-string line rewrites so comments survive; refuses (and changes nothing
in that file) if any expected line is missing or ambiguous — the configs are
textually identical in these blocks by construction (the effrainbow
migration enforced it).

    python experiments/src/update_100k_configs_bbf.py
"""
import pathlib
import sys

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
CONFIG = "config_fhrdqn_100k.yaml"

REPLACEMENTS = [
    (
        "  # Agent recipe: EfficientRainbowAgent = IQN-dueling + double + n-step +\n"
        "  # DrQ augmentation (src/agents/efficient_rainbow_agent.py); \"fhrdqn\"\n"
        "  # selects the legacy plain double-DQN FHRDQNAgent + NatureCNN.\n"
        "  agent_class: efficient_rainbow\n",
        "  # Agent recipe: BBFAgent = the BBF recipe (arXiv:2305.19452) on the\n"
        "  # IQN-dueling stack — Impala-CNN x4 + EMA target + n-step/gamma\n"
        "  # annealing + shrink-and-perturb resets + DrQ augmentation\n"
        "  # (src/agents/bbf_agent.py). \"efficient_rainbow\" selects the previous\n"
        "  # DrQ-recipe EfficientRainbowAgent + NatureCNNEncoder; \"fhrdqn\" the\n"
        "  # legacy plain double-DQN FHRDQNAgent + NatureCNN.\n"
        "  agent_class: bbf\n",
    ),
    (
        "  eps_min: 0.01\n"
        "  # eps 1 -> 0.01 over ~5k post-warm-up steps (DrQ): 0.01 ** (1/5000)\n"
        "  decay_rate: 0.999079\n",
        "  eps_min: 0.0\n"
        "  # bbf: LINEAR eps 1 -> 0 over agent.eps_decay_steps post-warm-up env\n"
        "  # steps; the exponential decay_rate below is IGNORED by this recipe.\n"
        "  decay_rate: 0.999079\n",
    ),
    (
        "  discount_factor: 0.99\n",
        "  discount_factor: 0.997            # BBF gamma END value (annealed from gamma_start)\n",
    ),
    (
        "  TD_LR: 1.0                        # hard target copy at every update tick\n",
        "  TD_LR: 1.0                        # unused by bbf: EMA target inside the agent\n",
    ),
    (
        "  grad_clip_norm: 10.0\n",
        "  grad_clip_norm: .inf              # BBF trains unclipped (NaN guard still active)\n",
    ),
    (
        "  # DrQ \"Efficient DQN\" recipe (arXiv:2004.13649 Table 4) + IQN head sizes:\n"
        "  n_step: 10                        # multi-step returns (sample-time aggregation)\n",
        "  # BBF recipe (official BBF.gin, RR=2 variant) + IQN head sizes:\n"
        "  n_step: 10                        # annealing START (sample-time aggregation)\n"
        "  n_step_final: 3                   # annealed 10 -> 3 over anneal_grad_steps\n"
        "  gamma_start: 0.97                 # annealed to discount_factor (0.997)\n"
        "  anneal_grad_steps: 10000          # annealing phase after each reset\n"
        "  target_ema_tau: 0.005             # EMA target per gradient step\n"
        "  target_action_selection: true     # act with the EMA target network (SR-SPR)\n"
        "  reset_interval_grad_steps: 40000  # shrink-and-perturb cadence (20k env steps at RR2)\n"
        "  no_resets_after_grad_steps: 130000 # skip the ~160k-grad-step reset: the final cycle must fully recover (no SPR/PER here)\n"
        "  shrink_factor: 0.5                # encoder <- 0.5*old + 0.5*random at each reset\n"
        "  perturb_factor: 0.5\n"
        "  eps_decay_steps: 2001             # linear eps decay length (env steps, BBF)\n",
    ),
    (
        "  head_hidden: 512\n",
        "  head_hidden: 2048                 # BBF hidden_dim\n",
    ),
    (
        "  prioritized_replay: false         # reserved: PER over the episodic buffer, later\n"
        "  # DrQ/Dopamine Adam settings (QAgent defaults keep the legacy wd=0.01):\n"
        "  weight_decay: 0.0\n",
        "  prioritized_replay: false         # deviation from BBF (PER): uniform episodic replay\n"
        "  # BBF AdamW: wd 0.1, masked to ndim>=2 params in-agent (biases exempt):\n"
        "  weight_decay: 0.1\n",
    ),
    (
        "network:\n"
        "  # fhrdqn recipe only (NatureCNN head width); the efficient_rainbow\n"
        "  # recipe builds its head from agent.head_hidden and ignores this.\n"
        "  fc_hidden: 512\n",
        "network:\n"
        "  # bbf recipe: Impala-CNN width multiplier (ImpalaCNNEncoder).\n"
        "  width_scale: 4\n"
        "  # fhrdqn recipe only (NatureCNN head width); efficient_rainbow and\n"
        "  # bbf build their heads from agent.head_hidden and ignore this.\n"
        "  fc_hidden: 512\n",
    ),
    (
        "  warmup_steps: 1600                # DER's min-replay-history\n",
        "  warmup_steps: 2000                # BBF's min-replay-history\n",
    ),
    (
        "  amsgrad: false\n",
        "  amsgrad: false\n"
        "  torch_compile: true               # compile the IQN quantile path (CUDA only)\n",
    ),
    (
        "  target_network_update_steps: 1     # DrQ: hard target copy every env step (TD_LR 1.0)\n",
        "  target_network_update_steps: 1     # no-op for bbf (EMA per gradient step in-agent)\n",
    ),
]


# Life handling: the 22 games with a lives counter switch from gymnasium's
# game-restarting terminal_on_life_loss to the Dopamine episodic-life
# protocol; the 5 no-lives games (Boxing/Freeway/Pong/PrivateEye/Enduro) just
# state episodic_life: false. Exactly one variant must match per config.
LIFE_VARIANTS = [
    (
        "    # Life loss ends the training episode (EfficientZero's setting, and this\n"
        "    # repo's existing Seaquest/Pac-Man convention).\n"
        "    terminal_on_life_loss: true\n",
        "    # Dopamine episodic-life protocol (what DER/DrQ/SPR/BBF train under):\n"
        "    # life loss ends the AGENT episode (bootstrap cut, EpisodicLifeWrapper)\n"
        "    # while the GAME continues to real game-over. Final evaluation always\n"
        "    # scores FULL games regardless (the launcher overrides both flags).\n"
        "    terminal_on_life_loss: false\n"
        "    episodic_life: true\n",
    ),
    (
        "    terminal_on_life_loss: false\n",
        "    terminal_on_life_loss: false\n"
        "    episodic_life: false           # no lives counter in this game\n",
    ),
]


def main():
    configs = sorted(ATARI.glob(f"dqn_*/{CONFIG}"))
    if len(configs) != 27:
        sys.exit(f"expected 27 game configs, found {len(configs)} — aborting")
    for path in configs:
        text = path.read_text()
        for old, new in REPLACEMENTS:
            count = text.count(old)
            if count != 1:
                sys.exit(f"{path}: expected exactly 1 occurrence of\n---\n{old}---\n"
                         f"found {count} — file left unchanged, aborting")
            text = text.replace(old, new)
        for old, new in LIFE_VARIANTS:
            if text.count(old) == 1:
                text = text.replace(old, new)
                break
        else:
            sys.exit(f"{path}: no life-handling variant matched — aborting")
        path.write_text(text)
        print(f"updated {path.relative_to(ATARI.parent)}")


if __name__ == "__main__":
    main()
