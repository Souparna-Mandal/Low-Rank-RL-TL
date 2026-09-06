"""Pack / unpack the run artefacts that are too large for git.

A run directory holds two kinds of files. The small reproducibility set
(config.yaml, rewards.csv, eval.csv, rank_stats.csv, window_hankel.csv,
hankel_sweep.csv, diag_binned_600.npz) is tracked in git, so every figure and
table of the results notebooks renders from a fresh checkout. The large
artefacts - checkpoints/final.pt (~5 MB each, needed for the rollout-Hankel
spectra and the videos) and videos/*.mp4 - are NOT in git; this script
bundles them per family so they can be attached to a GitHub release (or any
blob store) and restored on another machine:

    # on the machine that trained (from the experiment dir)
    python ../src/run_assets.py pack --manifest cached/sb3_runs_manifest_td3.json \
        --out ~/run_assets/sb3_ant_td3.tar.gz
    gh release upload td3-runs-v1 ~/run_assets/sb3_ant_td3.tar.gz   # once per family

    # on any other clone (from the same experiment dir)
    gh release download td3-runs-v1 --pattern 'sb3_ant_td3.tar.gz' --dir /tmp
    python ../src/run_assets.py unpack --archive /tmp/sb3_ant_td3.tar.gz

Paths inside the archive are relative to the experiment dir (cached/runs/...),
exactly as the manifest records them, so unpack lands the files where the
manifest and the notebooks expect them. --with-videos / --without-videos
controls the mp4s (default: included); --checkpoint picks final|best|latest.
"""
import argparse
import json
import pathlib
import tarfile


def _runs(manifest):
    m = json.load(open(manifest))
    for arm, seeds in m.get("runs", {}).items():
        for seed, rel in seeds.items():
            yield arm, seed, pathlib.Path(rel)


def pack(manifest, out, checkpoint="final", with_videos=True):
    out = pathlib.Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ck = n_vid = 0
    missing = []
    with tarfile.open(out, "w:gz") as tar:
        for arm, seed, rel in _runs(manifest):
            ck = rel / "checkpoints" / f"{checkpoint}.pt"
            if ck.exists():
                tar.add(ck, arcname=str(ck))
                n_ck += 1
            else:
                missing.append(f"{arm}/{seed}")
            if with_videos and (rel / "videos").is_dir():
                for mp4 in sorted((rel / "videos").glob("*.mp4")):
                    tar.add(mp4, arcname=str(mp4))
                    n_vid += 1
    size = out.stat().st_size / 1e6
    print(f"{out}: {n_ck} checkpoints, {n_vid} videos, {size:.0f} MB")
    if missing:
        print("no checkpoint for:", ", ".join(missing))
    return out


def unpack(archive, dest="."):
    dest = pathlib.Path(dest)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:      # refuse anything that would escape dest
            if m.name.startswith("/") or ".." in pathlib.Path(m.name).parts:
                raise ValueError(f"unsafe path in archive: {m.name}")
        tar.extractall(dest, filter="data")
    print(f"{archive}: {len(members)} files restored under {dest.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="final", choices=["final", "best", "latest"])
    p.add_argument("--without-videos", action="store_true")
    u = sub.add_parser("unpack")
    u.add_argument("--archive", required=True)
    u.add_argument("--dest", default=".")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.manifest, a.out, a.checkpoint, not a.without_videos)
    else:
        unpack(a.archive, a.dest)


if __name__ == "__main__":
    main()
