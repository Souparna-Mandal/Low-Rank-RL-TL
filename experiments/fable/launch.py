"""Sweep driver: run a plan of runner.py invocations with N parallel workers.

Plan file: JSON list of {variant, env, seed, episodes, set: [k=v,...]}.
Outputs land in <outdir>/<variant>__<env>__s<seed>.json; existing outputs are
skipped, so the sweep is resumable. Failures are logged and don't stop others.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from agents.variants import VARIANTS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNNER = str(REPO / "experiments" / "fable" / "runner.py")

# The interpreter running this script, so the sweep uses whatever environment
# was used to launch it (venv, conda, system) instead of assuming a .venv/
# layout. PPO_LAUNCH_PYTHON overrides it for cross-interpreter runs.
PY = os.environ.get("PPO_LAUNCH_PYTHON") or sys.executable

# Plan files are developer-authored, but they are still data that reaches an
# argv, so each field is checked against what runner.py will accept before it
# gets there: unknown variant names, odd env ids or non-integer seeds fail here
# with a clear message rather than downstream.
ENV_RE = re.compile(r"\A[A-Za-z0-9_/.-]+\Z")
SET_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.+-]+\Z")


def validate(spec):
    """Return the spec's fields as trusted, typed values, or raise ValueError."""
    variant = str(spec["variant"])
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")
    env_name = str(spec["env"])
    if not ENV_RE.match(env_name):
        raise ValueError(f"suspicious env id {env_name!r}")
    seed, episodes = int(spec["seed"]), int(spec["episodes"])
    sets = [str(s) for s in (spec.get("set") or [])]
    for s in sets:
        if not SET_RE.match(s):
            raise ValueError(f"malformed --set entry {s!r}; expected KEY=VALUE")
    return variant, env_name, seed, episodes, sets


def out_path(outdir, spec):
    return outdir / f"{spec['variant']}__{spec['env']}__s{spec['seed']}.json"


def run_one(spec, outdir):
    out = out_path(outdir, spec)
    if out.exists():
        return ("skip", str(out))
    try:
        variant, env_name, seed, episodes, sets = validate(spec)
    except (ValueError, KeyError, TypeError) as e:
        (outdir / "failures.log").open("a").write(f"\n=== {spec}\ninvalid spec: {e}\n")
        return ("FAIL", f"invalid spec: {e}")
    cmd = [PY, RUNNER, "--variant", variant, "--env", env_name,
           "--seed", str(seed), "--episodes", str(episodes), "--out", str(out)]
    if sets:
        cmd += ["--set", *sets]
    # shell=False (the default) with an argv list: arguments go straight to
    # execve with no shell parsing, so there is no command-injection surface
    # here, and every field was validated above. nosemgrep:
    # python.lang.security.audit.dangerous-subprocess-use-audit
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, shell=False,
                       env=env)
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
