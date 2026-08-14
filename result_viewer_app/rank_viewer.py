"""Local web viewer for RunLogger outputs.

Scans <exp>/runs/<run_id>/ and <exp>/cached/runs/<run_id>/ trees up to two
levels below root (root defaults to the repo's experiments/
directory) for the artifacts RunLogger writes — rank_stats.csv, rewards.csv,
hankel_sweep.csv, trajectories/*.npz, figures/*.png, config.yaml — and serves a
single-page app to step through the spectrum figures episode by episode, plot
rank metrics over training, and explore how low-rankness evolves over training
and persists across sub-trajectory lengths.

The viewer is deliberately decoupled from the training/experiment code: it
imports nothing from src/ or experiments/ and consumes only on-disk data
contracts (the run-dir artifact files above plus the optional
<exp>/cached/*manifest*.json a multi-experiment launcher writes — see
scan_manifests). Refactors elsewhere in the repo cannot break it unless those
file formats change.

Stdlib only, so it can be scp'd to the HPC and run against the runs there:

    python result_viewer_app/rank_viewer.py       # serves http://localhost:8501
    python result_viewer_app/rank_viewer.py --root /scratch/experiments --port 9000
"""

import argparse
import ast
import csv
import gzip
import json
import math
import pathlib
import re
import struct
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = pathlib.Path(__file__).resolve().parent
HTML_PATH = HERE / "rank_viewer.html"
FIG_RE = re.compile(r"^(?:ep(\d{6})|final)_(.+)\.png$")
TRAJ_RE = re.compile(r"^(?:ep(\d{6})|final)_seed(\d+)\.npz$")
# autoregressive_rollouts/epNNNNNN.npz — one file per probe checkpoint, holding
# actual vs one-step-ahead vs free-running arrays for a few example rollouts.
AR_ROLLOUT_RE = re.compile(r"^(?:ep(\d{6})|final)\.npz$")
# hankel_sweep.csv columns the frontend plots — the leverage *min* and matrix
# shape columns are dropped from the JSON payload (still in the raw CSV link).
# A long Atari run has >100k sweep rows, so payload size matters; we keep the
# scalar metrics the metric dropdown offers (incl. peak leverage = irs/ics max
# and sparsity) and nothing else.
SWEEP_COLS = ["episode", "matrix", "rollout", "seed", "sub_len",
              "eff_rank", "stable_rank", "spikiness",
              "nnz_rows", "nnz_cols",
              "row_coherence", "col_coherence",
              "row_lev_max", "col_lev_max"]


def _json_safe(v):
    """Recursively replace non-finite floats with None: json.loads accepts the
    non-standard NaN/Infinity tokens and json.dumps re-emits them, but the
    browser's JSON.parse rejects them outright."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return v


def _safe_text(path: pathlib.Path) -> str | None:
    """read_text that returns None instead of raising: unreadable/racing files
    (permissions, NFS blips on the HPC mounts this runs against) must cost the
    field, not the whole HTTP response."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


# ---- experiment manifests ---------------------------------------------------
# Multi-experiment launchers (e.g. experiments/src/run_fhrdqn_seeds.py) record
# their runs in <exp>/cached/*manifest*.json. The viewer treats that file as a
# DATA CONTRACT and never imports launcher code, so launcher/src refactors
# cannot break the viewer as long as the JSON keeps this shape (every field
# optional, unknown fields ignored, malformed files skipped):
#
#     {"seeds": [44, 66],
#      "runs": {"<arm>": {"<seed>": "<run dir, relative to the exp dir>"}},
#      "overrides": {"<arm>": {"<config param>": value}}}
#
# "arm" is a variant key ("baseline", "fhr", "exp3", ...); "overrides" is the
# config diff that arm actually trained with (applied on top of the config
# copy, which does NOT reflect it). A run dir listed here is the arm's CURRENT
# run for that seed — same-named dirs on disk that the manifest does not list
# are stale/foreign and must not be averaged into the arm.
def scan_manifests(root: pathlib.Path) -> dict[str, dict]:
    """exp (relative dir, str) ->
        {"families": {family: {"arms": {arm: {seed: run_dir_name}},
                               "overrides": {arm: {param: value}}}},
         "by_run": {run_dir_name: {"arm": ..., "seed": ..., "family": ...}}}

    Each manifest FILE is its own family (id = filename stem minus the
    "manifest" token: fhrdqn_runs_manifest.json -> "fhrdqn_runs",
    fhrdqn_runs_manifest.old-lambda0.5.json -> "fhrdqn_runs.old-lambda0.5").
    Families never merge: an archived family's "baseline" arm is NOT the
    current baseline, and folding them together would silently average runs
    from different experiment recipes."""
    out: dict[str, dict] = {}
    pats = ("*/cached/*manifest*.json", "*/*/cached/*manifest*.json")

    def mtime_ns(p):
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return 0
    # oldest file first, so when a run dir is listed in several manifest files
    # the most recently UPDATED file (the live one) wins the by_run attribution
    for path in sorted({p for pat in pats for p in root.glob(pat)},
                       key=lambda p: (mtime_ns(p), str(p))):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # torn mid-write or not JSON — skip, never crash the API
        if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
            continue
        exp = str(path.parent.parent.relative_to(root))
        family = path.stem.replace("_manifest", "").replace("manifest", "")
        family = family.strip("._-") or "runs"
        m = out.setdefault(exp, {"families": {}, "by_run": {}, "_files": {}})
        # distinct FILES must stay distinct families even when the derived id
        # collides (e.g. runs_manifest.json vs manifest.json both -> "runs")
        if family in m["_files"] and m["_files"][family] != path.name:
            family = path.stem
        m["_files"][family] = path.name
        fam = m["families"].setdefault(family, {"arms": {}, "overrides": {}})
        for arm, seeds in data["runs"].items():
            if not isinstance(seeds, dict):
                continue
            slots = fam["arms"].setdefault(str(arm), {})
            for seed, rel in seeds.items():
                if not isinstance(rel, str) or not rel.strip("/"):
                    continue
                name = rel.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                slots[str(seed)] = name
                m["by_run"][name] = {"arm": str(arm), "seed": str(seed),
                                     "family": family}
        ov = data.get("overrides")
        if isinstance(ov, dict):
            for arm, d in ov.items():
                if isinstance(d, dict):
                    # NaN/Infinity survive json.loads but are invalid JSON on
                    # re-serialisation — one such value would break every
                    # /api/runs poll in the browser's JSON.parse
                    fam["overrides"][str(arm)] = _json_safe(d)
    return out


def scan_runs(root: pathlib.Path, manifests: dict | None = None) -> list[dict]:
    """Every runs/<run_id> or cached/runs/<run_id> directory up to two levels
    below root that holds any artifact; exp is the experiment directory
    relative to root, e.g. "classical_control/dqn_cartpole". When the exp dir
    has a manifest (see scan_manifests), each run is annotated with its arm and
    seed ("tracked": true) or flagged "tracked": false so the frontend can keep
    stale dirs out of seed averages."""
    out = []
    patterns = ("*/runs/*/", "*/cached/runs/*/",
                "*/*/runs/*/", "*/*/cached/runs/*/")
    for stats in sorted({p for pat in patterns for p in root.glob(pat)}):
        artifacts = [p for p in (stats / "rank_stats.csv", stats / "rewards.csv",
                                 stats / "hankel_sweep.csv",
                                 stats / "train_diagnostics.csv")
                     if p.exists()]
        has_figs = (stats / "figures").is_dir() and any((stats / "figures").glob("*.png"))
        if not artifacts and not has_figs:
            continue
        exp_dir = stats.parent.parent
        if exp_dir.name == "cached":
            exp_dir = exp_dir.parent
        row = {
            "exp": str(exp_dir.relative_to(root)),
            "run": stats.name,
            "mtime": max(p.stat().st_mtime for p in artifacts) if artifacts
                     else stats.stat().st_mtime,
        }
        man = (manifests or {}).get(row["exp"])
        if man:
            info = man["by_run"].get(row["run"])
            if info:
                row["arm"], row["seed"] = info["arm"], info["seed"]
                row["family"] = info["family"]
                row["tracked"] = True
            else:
                # not (yet) in any manifest: still training, superseded by a
                # retrain, or written by something else — the frontend keeps it
                # out of tracked arm averages either way
                row["tracked"] = False
        out.append(row)
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _csv_table(path: pathlib.Path) -> dict | None:
    """CSV -> {columns, rows}, tolerating a torn final line / ragged rows since
    the training loop may be mid-append when we read (live runs)."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    header = rows[0]
    good = [r for r in rows[1:] if len(r) == len(header)]
    return {"columns": header, "rows": good}


def run_sig(run_dir: pathlib.Path) -> str:
    """Cheap change signature for a run: artifact file sizes/mtimes plus the
    figures/trajectories dir mtimes (which bump when a file is added). The
    frontend polls this instead of re-downloading the multi-MB payload."""
    parts = []
    for name in ("rank_stats.csv", "hankel_sweep.csv", "rewards.csv",
                 "train_diagnostics.csv",
                 "autoregressive_value_metrics.csv",
                 "autoregressive_value_coefficients.csv",
                 "autoregressive_value_horizon_metrics.csv"):
        p = run_dir / name
        if p.exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    for name in ("figures", "trajectories", "autoregressive_rollouts"):
        p = run_dir / name
        if p.is_dir():
            parts.append(f"{name}:{p.stat().st_mtime_ns}")
    return "|".join(parts)


# Envs whose reward is ±1 per step, so |episode reward| recovers the episode
# length from a legacy two-column rewards.csv: exact for CartPole (+1/step)
# and MountainCar (-1/step), within one step on Acrobot (terminal step pays 0).
DERIVABLE_STEPS_ENVS = ("CartPole", "MountainCar", "Acrobot")


def _env_name_and_episodic(run_dir: pathlib.Path) -> tuple[str | None, bool]:
    """(environment.name, rows-are-episodes?) from the run's config.yaml copy,
    stdlib-only. A policy-iteration config (training.no_iterations) writes one
    rewards.csv row per PI iteration whose reward is a mean greedy return —
    |reward| is not an episode length there, so steps must not be derived."""
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None, False
    text = cfg.read_text(errors="replace")
    name, in_env = None, False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            in_env = line.startswith("environment:")
            continue
        if in_env:
            m = re.match(r"\s+name:\s*([^\s#]+)", line)
            if m:
                name = m.group(1).strip("'\"")
                break
    return name, "no_episodes" in text


def _read_rewards(run_dir: pathlib.Path) -> tuple[list | None, str | None]:
    """rewards.csv -> ([[episode, reward, steps|None], ...], steps_source).
    steps_source is "logged" when the CSV has a steps column, "derived" when a
    legacy two-column file belongs to a ±1-reward-per-step env (steps filled in
    as |reward|), and None when no env-steps axis is possible."""
    path = run_dir / "rewards.csv"
    if not path.exists():
        return None, None
    pts = []
    has_steps = False
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        has_steps = "steps" in header
        si = header.index("steps") if has_steps else -1
        for row in reader:
            try:
                ep, rew = int(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue  # torn last line during a rewrite
            st = None
            if has_steps:
                try:
                    st = float(row[si])
                except (ValueError, IndexError):
                    pass  # torn steps cell — leave null, frontend skips it
            pts.append([ep, rew, st])
    if not pts:
        return pts, None
    if has_steps:
        return pts, "logged"
    env, episodic = _env_name_and_episodic(run_dir)
    if episodic and env and env.startswith(DERIVABLE_STEPS_ENVS):
        for p in pts:
            # a nan/inf reward cell has no derivable step count — leave it
            # None (round() would raise) like a torn steps cell
            if math.isfinite(p[1]):
                p[2] = abs(round(p[1]))
        return pts, "derived"
    return pts, None


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sig6(x: float) -> float:
    """Round to 6 significant digits so aggregate JSON stays compact."""
    return float(f"{x:.6g}")


def load_summary(run_dir: pathlib.Path) -> dict:
    """Compact per-run payload for the compare view: rewards, the (small)
    rank_stats table, the Hankel sweep collapsed to full-rollout-length means
    per (episode, matrix), and train_diagnostics collapsed to per-episode
    column means. No figures/trajectories — overlaying several seeds of
    several experiments must stay cheap over a tunnel."""
    rewards, steps_source = _read_rewards(run_dir)
    out = {"sig": run_sig(run_dir),
           "rewards": rewards, "steps_source": steps_source,
           # config copy text — the compare view diffs it across variants
           "config": _safe_text(run_dir / "config.yaml"),
           "stats": _csv_table(run_dir / "rank_stats.csv"),
           "sweep_evo": None, "diag": None}

    t = _csv_table(run_dir / "train_diagnostics.csv")
    if t and "episode" in t["columns"]:
        cols = [c for c in t["columns"] if c != "episode"]
        ei = t["columns"].index("episode")
        idx = [t["columns"].index(c) for c in cols]
        acc: dict[int, list] = {}
        for r in t["rows"]:
            e = _num(r[ei])
            if e is None:
                continue
            a = acc.setdefault(int(e), [[0.0, 0] for _ in cols])
            for j, ci in enumerate(idx):
                v = _num(r[ci])
                if v is not None:
                    a[j][0] += v
                    a[j][1] += 1
        rows = [[e] + [(_sig6(s / n) if n else None) for s, n in acc[e]]
                for e in sorted(acc)]
        out["diag"] = {"columns": ["episode"] + cols, "rows": rows}

    sweep = _csv_table(run_dir / "hankel_sweep.csv")
    if sweep:
        c = {k: i for i, k in enumerate(sweep["columns"])}
        if all(k in c for k in ("episode", "matrix", "sub_len")):
            metrics = [k for k in SWEEP_COLS if k in c and k not in
                       ("episode", "matrix", "rollout", "seed", "sub_len")]
            # full-rollout row = the longest sub_len per (episode, matrix, rollout)
            best: dict[tuple, tuple] = {}
            for r in sweep["rows"]:
                e, ln = _num(r[c["episode"]]), _num(r[c["sub_len"]])
                if e is None or ln is None:
                    continue
                key = (int(e), r[c["matrix"]],
                       r[c["rollout"]] if "rollout" in c else 0)
                if key not in best or ln > best[key][0]:
                    best[key] = (ln, r)
            # then mean across rollouts per (episode, matrix)
            acc2: dict[tuple, list] = {}
            for (e, m, _ro), (_ln, r) in best.items():
                a = acc2.setdefault((e, m), [[0.0, 0] for _ in metrics])
                for j, k in enumerate(metrics):
                    v = _num(r[c[k]])
                    if v is not None:
                        a[j][0] += v
                        a[j][1] += 1
            rows = [[e, m] + [(_sig6(s / n) if n else None) for s, n in a]
                    for (e, m), a in sorted(acc2.items())]
            out["sweep_evo"] = {"columns": ["episode", "matrix"] + metrics,
                                "rows": rows}
    return out


def load_run(run_dir: pathlib.Path) -> dict:
    """Parse one run directory into the JSON payload the frontend renders."""
    payload: dict = {"config": None, "stats": None, "sweep": None,
                     "rewards": None, "figures": {}, "trajectories": [],
                     "autoregressive_metrics": None,
                     "autoregressive_coefficients": None,
                     "autoregressive_horizons": None,
                     "autoregressive_rollouts": [],
                     "sig": run_sig(run_dir)}

    payload["config"] = _safe_text(run_dir / "config.yaml")

    payload["stats"] = _csv_table(run_dir / "rank_stats.csv")

    # Per-train() diagnostics (td_loss, penalty terms, learned recurrence
    # coefficients, ...) — written by agents whose train() returns a dict.
    payload["train_diagnostics"] = _csv_table(run_dir / "train_diagnostics.csv")

    # Autoregressive value-recurrence probe. Both tables are small (a handful
    # of rows per checkpoint) so they ship whole rather than being projected
    # down like the Hankel sweep.
    payload["autoregressive_metrics"] = _csv_table(
        run_dir / "autoregressive_value_metrics.csv")
    payload["autoregressive_coefficients"] = _csv_table(
        run_dir / "autoregressive_value_coefficients.csv")
    payload["autoregressive_horizons"] = _csv_table(
        run_dir / "autoregressive_value_horizon_metrics.csv")
    ar_rollout_dir = run_dir / "autoregressive_rollouts"
    if ar_rollout_dir.is_dir():
        for p in sorted(ar_rollout_dir.glob("*.npz")):
            m = AR_ROLLOUT_RE.match(p.name)
            if m:
                episode = m.group(1)
                payload["autoregressive_rollouts"].append({
                    "file": p.name,
                    "episode": None if episode is None else int(episode),
                })
    sweep = _csv_table(run_dir / "hankel_sweep.csv")
    if sweep:  # project onto the plotted columns to keep the payload small
        idx = [sweep["columns"].index(c) for c in SWEEP_COLS
               if c in sweep["columns"]]
        payload["sweep"] = {
            "columns": [sweep["columns"][i] for i in idx],
            "rows": [[r[i] for i in idx] for r in sweep["rows"]],
        }

    trajdir = run_dir / "trajectories"
    if trajdir.is_dir():
        for p in sorted(trajdir.glob("*.npz")):
            m = TRAJ_RE.match(p.name)
            if m:
                ep, seed = m.groups()
                payload["trajectories"].append({
                    "file": p.name,
                    "episode": None if ep is None else int(ep),
                    "seed": int(seed),
                })

    payload["rewards"], payload["steps_source"] = _read_rewards(run_dir)

    figdir = run_dir / "figures"
    if figdir.is_dir():
        for p in sorted(figdir.glob("*.png")):
            m = FIG_RE.match(p.name)
            if not m:
                continue
            ep, slug = m.groups()
            entry = payload["figures"].setdefault(slug, {"episodes": [], "final": False})
            if ep is None:
                entry["final"] = True
            else:
                entry["episodes"].append(int(ep))
        for entry in payload["figures"].values():
            entry["episodes"].sort()
    return payload


_NPY_FMT = {"<f8": ("d", 8), "<f4": ("f", 4), "<i8": ("q", 8), "<i4": ("i", 4)}


def load_npz_1d(path: pathlib.Path) -> dict[str, list[float]]:
    """Read the 1-D numeric arrays out of an .npz without numpy (the viewer is
    stdlib-only so it can run on the HPC). An npz is a zip of .npy members; the
    .npy header is an ast-parsable dict followed by raw little-endian data."""
    out: dict[str, list[float]] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            raw = z.read(name)
            if raw[:6] != b"\x93NUMPY":
                continue
            major = raw[6]
            if major == 1:
                hlen = struct.unpack("<H", raw[8:10])[0]
                hstart = 10
            else:  # version 2/3 use a 4-byte header length
                hlen = struct.unpack("<I", raw[8:12])[0]
                hstart = 12
            header = ast.literal_eval(raw[hstart:hstart + hlen].decode("latin1"))
            fmt = _NPY_FMT.get(header.get("descr"))
            shape = header.get("shape", ())
            if fmt is None or header.get("fortran_order") or len(shape) != 1:
                continue  # only plain 1-D numeric arrays are plotted
            n = shape[0]
            ch, size = fmt
            data = raw[hstart + hlen:hstart + hlen + n * size]
            out[name[:-4]] = list(struct.unpack(f"<{n}{ch}", data))
    return out


def make_handler(root: pathlib.Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet per-request logging
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            # gzip compressible payloads (JSON/CSV/HTML); a long run's payload
            # is tens of MB raw, ~4-5x smaller gzipped — critical over an SSH
            # tunnel to the HPC. PNGs are already compressed, skip them.
            if (len(body) > 8192 and not ctype.startswith("image/")
                    and "gzip" in self.headers.get("Accept-Encoding", "")):
                body = gzip.compress(body, 5)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _run_dir(self, exp: str, run: str) -> pathlib.Path | None:
            """Resolve a run directory, refusing anything that escapes root."""
            for base in ("cached/runs", "runs"):
                d = (root / exp / base / run).resolve()
                if d.is_dir() and d.is_relative_to(root):
                    return d
            return None

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            parts = [p for p in path.split("/") if p]
            try:
                if path == "/":
                    self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/runs":
                    # {"runs": [...], "manifests": {exp: {families: ...}}} —
                    # the frontend also accepts a bare list (older backends)
                    manifests = scan_manifests(root)
                    self._json({
                        "runs": scan_runs(root, manifests),
                        "manifests": {exp: {"families": m["families"]}
                                      for exp, m in manifests.items()},
                    })
                # exp may contain slashes (e.g. classical_control/dqn_cartpole),
                # so routes parse from the end: run ids / filenames never do.
                elif len(parts) >= 4 and parts[:2] == ["api", "run"]:
                    d = self._run_dir("/".join(parts[2:-1]), parts[-1])
                    if d is None:
                        self._json({"error": "run not found"}, 404)
                    else:
                        self._json(load_run(d))
                elif len(parts) >= 4 and parts[:2] == ["api", "summary"]:
                    d = self._run_dir("/".join(parts[2:-1]), parts[-1])
                    if d is None:
                        self._json({"error": "run not found"}, 404)
                    else:
                        self._json(load_summary(d))
                elif len(parts) >= 4 and parts[:2] == ["api", "sig"]:
                    d = self._run_dir("/".join(parts[2:-1]), parts[-1])
                    if d is None:
                        self._json({"error": "run not found"}, 404)
                    else:
                        self._json({"sig": run_sig(d)})
                elif len(parts) >= 5 and parts[:2] == ["api", "traj"]:
                    d = self._run_dir("/".join(parts[2:-2]), parts[-2])
                    npz = (d / "trajectories" / parts[-1]).resolve() if d else None
                    if npz and npz.is_file() and npz.suffix == ".npz" \
                            and npz.is_relative_to(root):
                        self._json(load_npz_1d(npz))
                    else:
                        self._json({"error": "trajectory not found"}, 404)
                elif len(parts) >= 5 and parts[:2] == ["api", "arroll"]:
                    # One checkpoint's autoregressive example rollouts: the
                    # actual, one-step-ahead and free-running arrays keyed by
                    # order/split/subset/index.
                    d = self._run_dir("/".join(parts[2:-2]), parts[-2])
                    npz = ((d / "autoregressive_rollouts" / parts[-1]).resolve()
                           if d else None)
                    if npz and npz.is_file() and npz.suffix == ".npz" \
                            and npz.is_relative_to(root):
                        self._json(load_npz_1d(npz))
                    else:
                        self._json({"error": "rollout not found"}, 404)
                elif len(parts) >= 4 and parts[0] == "csv":
                    d = self._run_dir("/".join(parts[1:-2]), parts[-2])
                    f = (d / parts[-1]).resolve() if d else None
                    if f and f.is_file() and f.suffix == ".csv" \
                            and f.is_relative_to(root):
                        self._send(200, f.read_bytes(), "text/csv; charset=utf-8")
                    else:
                        self._send(404, b"not found", "text/plain")
                elif len(parts) >= 4 and parts[0] == "fig":
                    d = self._run_dir("/".join(parts[1:-2]), parts[-2])
                    fig = (d / "figures" / parts[-1]).resolve() if d else None
                    if fig and fig.is_file() and fig.suffix == ".png" \
                            and fig.is_relative_to(root):
                        self._send(200, fig.read_bytes(), "image/png")
                    else:
                        self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path,
                    default=HERE.parent / "experiments",
                    help="directory containing <exp>/runs/<run_id>/ or "
                         "<exp>/cached/runs/<run_id>/ trees (searched up to "
                         "two levels down)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"rank viewer on http://{args.host}:{args.port}  (root: {root})")
    server.serve_forever()


if __name__ == "__main__":
    main()
