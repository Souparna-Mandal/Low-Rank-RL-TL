"""Local web viewer for RunLogger outputs.

Scans <root>/*/runs/<run_id>/ (root defaults to the repo's experiments/
directory) for the artifacts RunLogger writes — rank_stats.csv, rewards.csv,
hankel_sweep.csv, trajectories/*.npz, figures/*.png, config.yaml — and serves a
single-page app to step through the spectrum figures episode by episode, plot
rank metrics over training, and explore how low-rankness evolves over training
and persists across sub-trajectory lengths.

Stdlib only, so it can be scp'd to the HPC and run against the runs there:

    python result_viewer_app/rank_viewer.py       # serves http://localhost:8501
    python result_viewer_app/rank_viewer.py --root /scratch/experiments --port 9000
"""

import argparse
import ast
import csv
import gzip
import json
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


def scan_runs(root: pathlib.Path) -> list[dict]:
    """Every <exp>/runs/<run_id> directory under root that holds any artifact."""
    out = []
    for stats in sorted(root.glob("*/runs/*/")):
        artifacts = [p for p in (stats / "rank_stats.csv", stats / "rewards.csv",
                                 stats / "hankel_sweep.csv")
                     if p.exists()]
        has_figs = (stats / "figures").is_dir() and any((stats / "figures").glob("*.png"))
        if not artifacts and not has_figs:
            continue
        out.append({
            "exp": stats.parent.parent.name,
            "run": stats.name,
            "mtime": max(p.stat().st_mtime for p in artifacts) if artifacts
                     else stats.stat().st_mtime,
        })
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
    for name in ("rank_stats.csv", "hankel_sweep.csv", "rewards.csv"):
        p = run_dir / name
        if p.exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    for name in ("figures", "trajectories"):
        p = run_dir / name
        if p.is_dir():
            parts.append(f"{name}:{p.stat().st_mtime_ns}")
    return "|".join(parts)


def load_run(run_dir: pathlib.Path) -> dict:
    """Parse one run directory into the JSON payload the frontend renders."""
    payload: dict = {"config": None, "stats": None, "sweep": None,
                     "rewards": None, "figures": {}, "trajectories": [],
                     "sig": run_sig(run_dir)}

    cfg = run_dir / "config.yaml"
    if cfg.exists():
        payload["config"] = cfg.read_text(errors="replace")

    payload["stats"] = _csv_table(run_dir / "rank_stats.csv")
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

    rewards = run_dir / "rewards.csv"
    if rewards.exists():
        pts = []
        with open(rewards, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                try:
                    pts.append([int(row[0]), float(row[1])])
                except (ValueError, IndexError):
                    continue  # torn last line during a rewrite
        payload["rewards"] = pts

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
            d = (root / exp / "runs" / run).resolve()
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
                    self._json(scan_runs(root))
                elif len(parts) == 4 and parts[:2] == ["api", "run"]:
                    d = self._run_dir(parts[2], parts[3])
                    if d is None:
                        self._json({"error": "run not found"}, 404)
                    else:
                        self._json(load_run(d))
                elif len(parts) == 4 and parts[:2] == ["api", "sig"]:
                    d = self._run_dir(parts[2], parts[3])
                    if d is None:
                        self._json({"error": "run not found"}, 404)
                    else:
                        self._json({"sig": run_sig(d)})
                elif len(parts) == 5 and parts[:2] == ["api", "traj"]:
                    d = self._run_dir(parts[2], parts[3])
                    npz = (d / "trajectories" / parts[4]).resolve() if d else None
                    if npz and npz.is_file() and npz.suffix == ".npz" \
                            and npz.is_relative_to(root):
                        self._json(load_npz_1d(npz))
                    else:
                        self._json({"error": "trajectory not found"}, 404)
                elif len(parts) == 4 and parts[0] == "csv":
                    d = self._run_dir(parts[1], parts[2])
                    f = (d / parts[3]).resolve() if d else None
                    if f and f.is_file() and f.suffix == ".csv" \
                            and f.is_relative_to(root):
                        self._send(200, f.read_bytes(), "text/csv; charset=utf-8")
                    else:
                        self._send(404, b"not found", "text/plain")
                elif len(parts) == 4 and parts[0] == "fig":
                    d = self._run_dir(parts[1], parts[2])
                    fig = (d / "figures" / parts[3]).resolve() if d else None
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
                    help="directory containing <exp>/runs/<run_id>/ trees")
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
