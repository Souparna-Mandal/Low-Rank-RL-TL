"""Rebuild Atari-100k manifests from the run directories on disk.

Cluster (child-mode) runs never touch a manifest — the PBS array launches one
`run_fhrdqn_atari100k.py --arm ...` process per (game, arm, seed) and the
manifest bookkeeping normally done by launch_all() is skipped. After syncing
cached/runs/ back from CX3, run this to reconstruct
cached/<family>_runs_manifest.json per game dir so the notebooks and the
result viewer load the runs exactly as if launch_all had trained them.

A run counts when its dir matches <experiment.name>_<arm>_seed<S>_<timestamp>
and contains eval_summary.json (i.e. it finished); the NEWEST run per
(arm, seed) wins. seeds come from the config; overrides from its
fhr_experiments block.

    python experiments/src/rebuild_atari_manifests.py                    # 100k family
    python experiments/src/rebuild_atari_manifests.py --family tune      # tuning family
    python experiments/src/rebuild_atari_manifests.py --games boxing seaquest
"""
import argparse
import json
import pathlib
import re

import yaml

ATARI = pathlib.Path(__file__).resolve().parents[1] / "atari"
FAMILIES = {
    # family key -> (config file, manifest file name)
    "100k": ("config_effrainbow_100k.yaml", "effrainbow100k_runs_manifest.json"),
    "tune": ("config_effrainbow_tune.yaml", "effrainbowtune_runs_manifest.json"),
}


def rebuild(game_dir: pathlib.Path, config: str, manifest_name: str) -> dict | None:
    cfg_path = game_dir / config
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    name = cfg["experiment"]["name"]
    seeds = list(cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]])
    sweeps = cfg["experiment"].get("fhr_experiments") or {}
    overrides = {"baseline": {"fhr_weight": 0.0},
                 **{f"exp{n}": ov for n, ov in sweeps.items()}}

    pat = re.compile(re.escape(name)
                     + r"_(baseline|exp\d+)_seed(\d+)_(\d{8}-\d{6})$")
    runs: dict[str, dict[str, tuple[str, str]]] = {}
    for d in sorted((game_dir / "cached/runs").glob(f"{name}_*")):
        m = pat.match(d.name)
        if not m or not (d / "eval_summary.json").exists():
            continue
        arm, seed, ts = m.groups()
        best = runs.setdefault(arm, {}).get(seed)
        if best is None or ts > best[1]:            # newest run per (arm, seed)
            runs[arm][seed] = (f"cached/runs/{d.name}", ts)
    if not runs:
        return None
    manifest = {"seeds": seeds,
                "runs": {arm: {s: rel for s, (rel, _) in sorted(by_seed.items())}
                         for arm, by_seed in sorted(runs.items())},
                "overrides": {k: overrides[k] for k in runs if k in overrides}}
    out = game_dir / "cached" / manifest_name
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), default="100k")
    ap.add_argument("--games", nargs="*", default=None,
                    help="game dir stems, e.g. boxing bank_heist (default all)")
    args = ap.parse_args()
    config, manifest_name = FAMILIES[args.family]
    dirs = (sorted(ATARI.glob("dqn_*")) if args.games is None
            else [ATARI / f"dqn_{g}" for g in args.games])
    for game_dir in dirs:
        manifest = rebuild(game_dir, config, manifest_name)
        if manifest is None:
            continue
        n = sum(len(v) for v in manifest["runs"].values())
        print(f"{game_dir.name:24s} {n:3d} run(s) -> cached/{manifest_name} "
              f"arms={sorted(manifest['runs'])}")


if __name__ == "__main__":
    main()
