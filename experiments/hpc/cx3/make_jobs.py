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
    # notebook-comparison protocol (no checkpoints, episode-gated analysis);
    # seeds = the global yaml's ref_seeds (seed 0 exists from the notebook pass):
    "ref": ("config_effrainbow_100k_ref.yaml", ["baseline", "exp3"]),
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
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write here instead of jobs_<mode>.txt (use for a "
                         "resubmission list while an earlier array is still "
                         "queued on jobs_<mode>.txt — never rewrite a file a "
                         "live array reads by line number)")
    ap.add_argument("--exclude-lines", default="",
                    help="comma-separated 1-based line numbers of jobs_<mode>.txt "
                         "to leave out (runs still queued/running in a live array)")
    args = ap.parse_args()
    config, arm_filter = MODES[args.mode]
    live = set()
    if args.exclude_lines:
        base = (HERE / f"jobs_{args.mode}.txt").read_text().splitlines()
        live = {base[int(n) - 1] for n in args.exclude_lines.split(",") if n}

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
                line = f"{game_dir.name}\t{config}\t{arm_key}\t{seed}\t{ov}"
                if line in live:
                    skipped += 1
                    continue
                lines.append(line)

    out = (args.out or HERE / f"jobs_{args.mode}.txt").resolve()
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(lines)} job(s) -> {out}"
          + (f" ({skipped} already finished, skipped)" if skipped else ""))
    if lines:
        packs = -(-len(lines) // 2)          # default PACK=2 (2 runs per GPU)
        v = f"-v JOBS_FILE={out.relative_to(HERE.parents[2])} " if args.out else ""
        print(f"submit (PACK=2 default): from experiments/hpc/cx3/logs run\n"
              f"  qsub {v}-J 1-{packs}%12 ../{args.mode}.pbs\n"
              f"(PACK=1: qsub {v}-v PACK=1 -J 1-{len(lines)}%12 ../{args.mode}.pbs)")


if __name__ == "__main__":
    main()
