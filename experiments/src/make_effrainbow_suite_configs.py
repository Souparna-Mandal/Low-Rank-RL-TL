"""Generate config_effrainbow_100k.yaml for every Atari-100k game dir from the
Krull template (the burst-cadence EfficientRainbow recipe + the exp3 FHR arm).

Per game, the template is specialised with: the experiment name, the ALE env
id, episodic_life (false for the no-lives games — read from that game's bbf
config, the source of truth), include_in_aggregate, and single-seed [0] for
the suite's first pass (Krull keeps its own existing config untouched).

    python experiments/src/make_effrainbow_suite_configs.py
"""
import pathlib
import re
import sys

import yaml

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
TEMPLATE = ATARI / "dqn_krull" / "config_effrainbow_100k.yaml"


def main():
    template = TEMPLATE.read_text()
    made = 0
    for bbf_cfg_path in sorted(ATARI.glob("dqn_*/config_fhrdqn_100k.yaml")):
        game_dir = bbf_cfg_path.parent
        if game_dir.name == "dqn_krull":
            continue
        bbf = yaml.safe_load(bbf_cfg_path.read_text())
        text = template
        # experiment name + env id
        text = text.replace("  name: dqn_krull_effrainbow100k",
                            f"  name: {game_dir.name}_effrainbow100k")
        text = text.replace("  name: ALE/Krull-v5",
                            f"  name: {bbf['environment']['name']}")
        # single-seed first pass
        text = text.replace("  seeds: [0, 1]", "  seeds: [0]")
        # aggregate membership follows the game's bbf config
        if not bbf["experiment"].get("include_in_aggregate", False):
            text = text.replace("  include_in_aggregate: true",
                                "  include_in_aggregate: false")
        # lives handling follows the game's bbf config
        if not bbf["environment"]["atari"].get("episodic_life", False):
            text = text.replace("    episodic_life: true",
                                "    episodic_life: false  # no lives counter in this game")
        # drop any commented-out experiment leftovers from the template
        text = re.sub(r"\n( *# (?:ARX:|\(see|the ONLY|Paired).*| *# \d+: \{.*)",
                      "", text)
        out = game_dir / "config_effrainbow_100k.yaml"
        out.write_text(text)
        made += 1
        print(f"wrote {out.relative_to(ATARI.parent)}")
    if made != 26:
        sys.exit(f"expected 26 generated configs, wrote {made}")


if __name__ == "__main__":
    main()
