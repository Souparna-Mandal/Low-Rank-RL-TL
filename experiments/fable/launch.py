"""Sweep driver: run a plan of runner.py invocations with N parallel workers.

Plan file: JSON list of {variant, env, seed, episodes, set: [k=v,...]}.
Outputs land in <outdir>/<variant>__<env>__s<seed>.json; existing outputs are
skipped, so the sweep is resumable. Failures are logged and don't stop others.
"""
import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = pathlib.Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python")
RUNNER = str(REPO / "experiments" / "fable" / "runner.py")


def out_path(outdir, spec):
    return outdir / f"{spec['variant']}__{spec['env']}__s{spec['seed']}.json"


def run_one(spec, outdir):
    out = out_path(outdir, spec)
    if out.exists():
        return ("skip", str(out))
    cmd = [PY, RUNNER, "--variant", spec["variant"], "--env", spec["env"],
           "--seed", str(spec["seed"]), "--episodes", str(spec["episodes"]),
           "--out", str(out)]
    if spec.get("set"):
        cmd += ["--set", *spec["set"]]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        (outdir / "failures.log").open("a").write(
            f"\n=== {spec}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}\n")
        return ("FAIL", str(spec))
    return ("ok", r.stdout.strip().splitlines()[-1] if r.stdout else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    plan = json.loads(pathlib.Path(a.plan).read_text())
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    done = fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, s, outdir): s for s in plan}
        for f in as_completed(futs):
            status, msg = f.result()
            done += status in ("ok", "skip")
            fail += status == "FAIL"
            print(f"[{done + fail}/{len(plan)}] {status} {msg}", flush=True)
    print(f"sweep complete: {done} ok/skipped, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
