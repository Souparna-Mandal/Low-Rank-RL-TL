"""Launch every (or the named) tuning variant's runs, one variant at a time.

    cd experiments/stable_baselines_3/mountaincar_tuning
    python launch_variants.py                    # all variants in VARIANTS
    python launch_variants.py cad2_1 cad4_2      # just these
    python launch_variants.py --max-workers 6

Within a variant the usual run_sb3_seeds fan-out applies (baseline + exp arms
x seeds, manifest skip/resume). Variants run sequentially so the box never
holds more than max-workers trainings at once.
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from make_variants import VARIANTS                     # noqa: E402
from run_sb3_seeds import launch_all                   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--experiment", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = args.variants or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown} — defined: {list(VARIANTS)}")
    for name in names:
        vdir = HERE / name
        if not (vdir / "configs" / "config_sb3.yaml").exists():
            raise SystemExit(f"{vdir} has no config — run make_variants.py first")
        print(f"=== variant {name} ===")
        launch_all(max_workers=args.max_workers, force=args.force,
                   exp_dir=vdir, experiments=args.experiment)


if __name__ == "__main__":
    main()
