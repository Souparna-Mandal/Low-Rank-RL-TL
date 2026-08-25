"""One-off migration: switch all 27 Atari-100k game configs to the
EfficientRainbowAgent recipe (IQN-dueling + double + n-step 10 + DrQ
augmentation, DrQ data-efficient hyperparameters).

Exact-string line rewrites so comments survive; refuses (and changes nothing
in that file) if any expected line is missing or ambiguous — the configs are
textually identical in these blocks by construction.

    python experiments/src/update_100k_configs_effrainbow.py
"""
import pathlib
import sys

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
CONFIG = "config_fhrdqn_100k.yaml"

REPLACEMENTS = [
    (
        "  device: auto\n"
        "  save_artifacts: true\n",
        "  device: auto\n"
        "  save_artifacts: true\n"
        "  # Agent recipe: EfficientRainbowAgent = IQN-dueling + double + n-step +\n"
        "  # DrQ augmentation (src/agents/efficient_rainbow_agent.py); \"fhrdqn\"\n"
        "  # selects the legacy plain double-DQN FHRDQNAgent + NatureCNN.\n"
        "  agent_class: efficient_rainbow\n",
    ),
    (
        "  # eps 1 -> 0.01 over ~20k post-warm-up steps: 0.01 ** (1/20000)\n"
        "  decay_rate: 0.99977\n",
        "  # eps 1 -> 0.01 over ~5k post-warm-up steps (DrQ): 0.01 ** (1/5000)\n"
        "  decay_rate: 0.999079\n",
    ),
    (
        "  gd_steps_ceil: 1                  # with train_frequency_steps 1 -> replay ratio 1\n",
        "  gd_steps_ceil: 2                  # with train_frequency_steps 1 -> replay ratio 2\n",
    ),
    (
        "  warmup_grad_steps: 5000           # of ~98k total grad steps at replay ratio 1\n",
        "  warmup_grad_steps: 10000          # ~5k env steps of penalty warm-up at replay ratio 2\n",
    ),
    (
        "  double: true\n",
        "  double: true\n"
        "  # DrQ \"Efficient DQN\" recipe (arXiv:2004.13649 Table 4) + IQN head sizes:\n"
        "  n_step: 10                        # multi-step returns (sample-time aggregation)\n"
        "  n_quantiles: 8                    # online loss taus (also Double-Q selection)\n"
        "  n_quantiles_target: 8             # target loss taus\n"
        "  n_quantiles_act: 32               # fixed-grid taus for acting / analysis traces\n"
        "  n_quantiles_fhr: 8                # fixed-grid taus for the FHR anchor/lag forward\n"
        "  n_cos: 64\n"
        "  head_hidden: 512\n"
        "  huber_kappa: 1.0\n"
        "  use_augmentation: true            # DrQ random shift + intensity, gradient steps only\n"
        "  aug_pad: 4\n"
        "  aug_intensity: 0.05\n"
        "  prioritized_replay: false         # reserved: PER over the episodic buffer, later\n"
        "  # DrQ/Dopamine Adam settings (QAgent defaults keep the legacy wd=0.01):\n"
        "  weight_decay: 0.0\n"
        "  adam_eps: 0.00015\n"
        "  amsgrad: false\n",
    ),
    (
        "  target_network_update_steps: 1000\n",
        "  target_network_update_steps: 1     # DrQ: hard target copy every env step (TD_LR 1.0)\n",
    ),
    (
        "network:\n"
        "  # Nature DQN CNN (src/agents/atari_networks.py NatureCNN)\n"
        "  fc_hidden: 512\n",
        "network:\n"
        "  # fhrdqn recipe only (NatureCNN head width); the efficient_rainbow\n"
        "  # recipe builds its head from agent.head_hidden and ignores this.\n"
        "  fc_hidden: 512\n",
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
        path.write_text(text)
        print(f"updated {path.relative_to(ATARI.parent)}")


if __name__ == "__main__":
    main()
