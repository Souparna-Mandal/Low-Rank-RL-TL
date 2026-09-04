"""Render the GLOBAL EfficientRainbow Atari-100k config into every game dir.

Single source of truth: experiments/atari/config_effrainbow_100k.global.yaml.
Per game the __TOKEN__ placeholders are substituted with that game's fields
(read from its config_fhrdqn_100k.yaml, the stable source the suite generator
already used): env id, episodic_life (false for the no-lives games) and
include_in_aggregate. Comments survive because substitution is textual.

Also renders config_effrainbow_tune.yaml into the TUNE_GAMES dirs — the
hyperparameter-tuning variant for the CX3 campaign: 50k env steps, denser
checkpoints, seeds [0, 1]. The tuning ARM GRID lives in the global yaml
itself (experiment.tune_fhr_experiments — edit it THERE): this script strips
that block from the 100k renderings and swaps it in as the tune configs'
fhr_experiments.

    python experiments/src/sync_effrainbow_configs_from_global.py
"""
import pathlib
import re
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

# The tune grid + its comment banner in the global yaml, stripped from every
# 100k rendering and swapped into the tune configs as their fhr_experiments.
TUNE_BLOCK_RE = re.compile(                 # line-anchored on purpose: the
    r"  # ---- TUNING GRID.*\n"             # banner line,
    r"(?:  #.*\n)*"                         # its remaining comment lines,
    r"  tune_fhr_experiments:\n"            # the key,
    r"((?:    .*\n)+)")                     # -> group(1): the entry lines


def render(template: str, name: str, env_id: str, episodic_life: bool,
           include_in_aggregate: bool) -> str:
    out = TUNE_BLOCK_RE.sub("", template)
    out = (out
           .replace("__EXPERIMENT_NAME__", name)
           .replace("__ENV_ID__", env_id)
           .replace("__INCLUDE_IN_AGGREGATE__",
                    "true" if include_in_aggregate else "false")
           .replace("__EPISODIC_LIFE__",
                    "true" if episodic_life
                    else "false  # no lives counter in this game"))
    _validate(out)
    return out


def _validate(text: str) -> None:
    """Guard against a bad template edit or regex slip: every placeholder
    substituted, valid YAML, and every top-level section still present."""
    for tok in ("__EXPERIMENT_NAME__", "__ENV_ID__",
                "__INCLUDE_IN_AGGREGATE__", "__EPISODIC_LIFE__"):
        assert tok not in text, f"unsubstituted {tok} in rendered config"
    cfg = yaml.safe_load(text)
    for key in ("experiment", "environment", "network", "agent", "training",
                "evaluation", "analysis"):
        assert key in cfg, f"rendered config lost its {key}: section"
    assert "tune_fhr_experiments" not in cfg["experiment"]


def tune_variant(text: str, grid: str) -> str:
    """The 50k-step tuning rendering of an already-rendered game config.
    grid: the tune_fhr_experiments entry lines lifted from the global yaml,
    re-indented to sit under fhr_experiments."""
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
    text = text.replace(old_arm, grid.rstrip("\n"))
    _validate(text)
    return text


def main():
    template = GLOBAL.read_text()
    m = TUNE_BLOCK_RE.search(template)
    if m is None:
        sys.exit("no tune_fhr_experiments block in the global yaml")
    grid = m.group(1)
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
                tune_variant(text, grid))
            tuned += 1
        print(f"wrote {game_dir.name}"
              + (" (+ tune config)" if game_dir.name in TUNE_GAMES else ""))
    if made != 27 or tuned != len(TUNE_GAMES):
        sys.exit(f"expected 27 configs + {len(TUNE_GAMES)} tune configs, "
                 f"wrote {made} + {tuned}")


if __name__ == "__main__":
    main()
