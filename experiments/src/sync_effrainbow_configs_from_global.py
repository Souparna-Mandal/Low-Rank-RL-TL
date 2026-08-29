"""Render the GLOBAL EfficientRainbow Atari-100k config into every game dir.

Single source of truth: experiments/atari/config_effrainbow_100k.global.yaml.
Per game the __TOKEN__ placeholders are substituted with that game's fields
(read from its config_fhrdqn_100k.yaml, the stable source the suite generator
already used): env id, episodic_life (false for the no-lives games) and
include_in_aggregate. Comments survive because substitution is textual.

Also renders config_effrainbow_tune.yaml into the TUNE_GAMES dirs — the
hyperparameter-tuning variant for the CX3 campaign: 50k env steps, denser
checkpoints, seeds [0, 1], and the tuning fhr_experiments grid around the
suite arm (lambda 2, r 8). EDIT the grid below before submitting if you want
different arms — every entry is one (arm x game x seed) job on the cluster.

    python experiments/src/sync_effrainbow_configs_from_global.py
"""
import pathlib
import sys

import yaml

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
GLOBAL = ATARI / "config_effrainbow_100k.global.yaml"

# Tuning subset — picked from the completed 1-seed suite's per-game dHNS
# (exp3 - baseline): two big wins, two true washes with healthy baselines,
# and the worst loss, so the tuned hyperparameters must survive all three
# regimes rather than overfit the wins. Krull is deliberately EXCLUDED (it
# was the hand-tuning testbed already).
TUNE_GAMES = {
    "dqn_boxing":         "+1.47 dHNS — biggest win",
    "dqn_bank_heist":     "+0.62 dHNS — win from a weak baseline",
    "dqn_kung_fu_master": "+0.01 dHNS — wash, mid baseline",
    "dqn_battle_zone":    "+0.01 dHNS — wash, mid baseline",
    "dqn_demon_attack":   "-0.33 dHNS — worst loss",
}

# The tuning grid (arm number -> FHR overrides). 3 is the suite anchor; keep
# numbers >= 10 for new arms so they never collide with historical manifests.
TUNE_EXPERIMENTS = """\
    3:  {fhr_weight: 2, fhr_order: 8}                        # suite anchor
    10: {fhr_weight: 1, fhr_order: 8}
    11: {fhr_weight: 4, fhr_order: 8}
    12: {fhr_weight: 2, fhr_order: 4}
    13: {fhr_weight: 2, fhr_order: 16}
    14: {fhr_weight: 2, fhr_order: 8, reward_lags: True}     # ARX twin
    15: {fhr_weight: 2, fhr_order: 8, c_learning_rate: 0.001}"""


def render(template: str, name: str, env_id: str, episodic_life: bool,
           include_in_aggregate: bool) -> str:
    out = (template
           .replace("__EXPERIMENT_NAME__", name)
           .replace("__ENV_ID__", env_id)
           .replace("__INCLUDE_IN_AGGREGATE__",
                    "true" if include_in_aggregate else "false")
           .replace("__EPISODIC_LIFE__",
                    "true" if episodic_life
                    else "false  # no lives counter in this game"))
    yaml.safe_load(out)               # must render to valid YAML
    return out


def tune_variant(text: str) -> str:
    """The 50k-step tuning rendering of an already-rendered game config."""
    text = text.replace("_effrainbow100k", "_effrainbowtune")
    text = text.replace("  seeds: [0, 1, 2, 3, 4]", "  seeds: [0, 1]")
    text = text.replace(
        "  max_env_steps: 100000             # THE Atari-100k interaction budget",
        "  max_env_steps: 50000              # tuning budget: half of Atari-100k")
    text = text.replace(
        "  checkpoint_every_steps: 10000     # 10 tracking points across the 100k budget",
        "  checkpoint_every_steps: 5000      # 10 tracking points across the 50k budget")
    text = text.replace(
        "  step_freq: 10000                  # rank/Hankel tick every 10k ENV STEPS (at",
        "  step_freq: 5000                   # rank/Hankel tick every 5k ENV STEPS (at")
    old_arm = "    3: {fhr_weight: 2, fhr_order: 8}   # the suite arm (also the BBF comparison)"
    assert old_arm in text
    text = text.replace(old_arm, TUNE_EXPERIMENTS)
    yaml.safe_load(text)
    return text


def main():
    template = GLOBAL.read_text()
    made = tuned = 0
    for src_cfg in sorted(ATARI.glob("dqn_*/config_fhrdqn_100k.yaml")):
        game_dir = src_cfg.parent
        src = yaml.safe_load(src_cfg.read_text())
        text = render(
            template,
            name=f"{game_dir.name}_effrainbow100k",
            env_id=src["environment"]["name"],
            episodic_life=src["environment"]["atari"].get("episodic_life", True),
            include_in_aggregate=src["experiment"].get("include_in_aggregate",
                                                       False),
        )
        (game_dir / "config_effrainbow_100k.yaml").write_text(text)
        made += 1
        if game_dir.name in TUNE_GAMES:
            (game_dir / "config_effrainbow_tune.yaml").write_text(
                tune_variant(text))
            tuned += 1
        print(f"wrote {game_dir.name}"
              + (" (+ tune config)" if game_dir.name in TUNE_GAMES else ""))
    if made != 27 or tuned != len(TUNE_GAMES):
        sys.exit(f"expected 27 configs + {len(TUNE_GAMES)} tune configs, "
                 f"wrote {made} + {tuned}")


if __name__ == "__main__":
    main()
