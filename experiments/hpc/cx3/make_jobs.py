"""Generate the CX3 job list: one line per (game, arm, seed) training run.

Everything is config-driven — arms and seeds come from the game configs the
sync script rendered, so the jobs file can never disagree with what a run
will actually train.

    python experiments/hpc/cx3/make_jobs.py --mode tune    # -> jobs_tune.txt
    python experiments/hpc/cx3/make_jobs.py --mode suite   # -> jobs_suite.txt
    python experiments/hpc/cx3/make_jobs.py --mode suite --skip-existing

Line format (tab-separated, consumed by run_one.sh via $PBS_ARRAY_INDEX):
    <game_dir>\t<config>\t<arm_key>\t<seed>\t<agent-overrides json or ->

--skip-existing drops (game, arm, seed) triples that already have a finished
run dir (eval_summary.json present) — use it to build a resubmission list
after a partial pass.
"""
import argparse
import json
import pathlib
import re

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ATARI = HERE.parents[1] / "atari"
MODES = {
    # mode -> (config file, restrict arms to these keys | None = all defined)
    "tune": ("config_effrainbow_tune.yaml", None),
    "suite": ("config_effrainbow_100k.yaml", ["baseline", "exp3"]),
}


def finished(game_dir: pathlib.Path, name: str, arm: str, seed: int) -> bool:
    pat = re.compile(re.escape(name) + f"_{arm}_seed{seed}_" + r"\d{8}-\d{6}$")
    runs = game_dir / "cached/runs"
    return runs.exists() and any(
        pat.match(d.name) and (d / "eval_summary.json").exists()
        for d in runs.iterdir())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    config, arm_filter = MODES[args.mode]

    lines, skipped = [], 0
    for cfg_path in sorted(ATARI.glob(f"dqn_*/{config}")):
        game_dir = cfg_path.parent
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg["experiment"]["name"]
        seeds = list(cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]])
        sweeps = cfg["experiment"].get("fhr_experiments") or {}
        arms = [("baseline", "-")] + [
            (f"exp{n}", json.dumps(ov, separators=(",", ":")))
            for n, ov in sorted((int(k), v) for k, v in sweeps.items())]
        if arm_filter is not None:
            arms = [(k, ov) for k, ov in arms if k in arm_filter]
        for arm_key, ov in arms:
            for seed in seeds:
                if args.skip_existing and finished(game_dir, name, arm_key, seed):
                    skipped += 1
                    continue
                lines.append(f"{game_dir.name}\t{config}\t{arm_key}\t{seed}\t{ov}")

    out = HERE / f"jobs_{args.mode}.txt"
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(lines)} job(s) -> {out}"
          + (f" ({skipped} already finished, skipped)" if skipped else ""))
    if lines:
        print(f"submit:  qsub -J 1-{len(lines)}%12 "
              f"experiments/hpc/cx3/{args.mode}.pbs")


if __name__ == "__main__":
    main()
