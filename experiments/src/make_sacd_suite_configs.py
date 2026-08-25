"""Generate config_sacd_100k.yaml for every Atari-100k game dir from the
Krull template (the SAC-Discrete recipe + its FHR/ccond/PER arms).

Per game, the template is specialised with: the experiment name, the ALE env
id, episodic_life (false for the no-lives games — read from that game's
fhrdqn config, the source of truth), include_in_aggregate, and single-seed
[0] for the suite's first pass (Krull keeps its own existing config
untouched). Mirrors make_effrainbow_suite_configs.py.

    python experiments/src/make_sacd_suite_configs.py
"""
import pathlib
import sys

import yaml

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
TEMPLATE = ATARI / "dqn_krull" / "config_sacd_100k.yaml"


def main():
    template = TEMPLATE.read_text()
    made = 0
    for ref_cfg_path in sorted(ATARI.glob("dqn_*/config_fhrdqn_100k.yaml")):
        game_dir = ref_cfg_path.parent
        if game_dir.name == "dqn_krull":
            continue
        ref = yaml.safe_load(ref_cfg_path.read_text())
        text = template
        # experiment name + env id
        text = text.replace("  name: dqn_krull_sacd100k",
                            f"  name: {game_dir.name}_sacd100k")
        text = text.replace("  name: ALE/Krull-v5",
                            f"  name: {ref['environment']['name']}")
        # single-seed first pass
        text = text.replace("  seeds: [0, 1]", "  seeds: [0]")
        # aggregate membership follows the game's fhrdqn config
        if not ref["experiment"].get("include_in_aggregate", False):
            text = text.replace("  include_in_aggregate: true",
                                "  include_in_aggregate: false")
        # lives handling follows the game's fhrdqn config
        if not ref["environment"]["atari"].get("episodic_life", False):
            text = text.replace(
                "    episodic_life: true",
                "    episodic_life: false  # no lives counter in this game")
        out = game_dir / "config_sacd_100k.yaml"
        out.write_text(text)
        made += 1
        print(f"wrote {out.relative_to(ATARI.parent)}")
    if made != 26:
        sys.exit(f"expected 26 generated configs, wrote {made}")


if __name__ == "__main__":
    main()
