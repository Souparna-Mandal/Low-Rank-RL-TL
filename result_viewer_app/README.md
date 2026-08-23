# Result viewer app

A small local web app for exploring `RunLogger` outputs. It scans `<exp>/runs/<run_id>/` directories and gives you a practical way to track how the **low-rank structure** of the Hankel and Q matrices changes over training, both episode by episode and across the full run.

The viewer reads the logger outputs in read-only mode, lives entirely outside `src/`, and does not touch the training code.

## Run it

```bash
python result_viewer_app/rank_viewer.py            # serves at http://localhost:8501, root = ./experiments
python result_viewer_app/rank_viewer.py --root /scratch/experiments --port 9000
```

Then open the printed URL in your browser. It uses only the Python standard library, so there is no `pip install` step and nothing to build. That also makes it easy to copy `rank_viewer.py` and `rank_viewer.html` to a remote machine or HPC system and point `--root` at the runs directory there.

Options:

- `--root PATH` — directory containing `<exp>/runs/<run_id>/` trees (default: `./experiments`)
- `--host HOST` — bind address (default: `127.0.0.1`)
- `--port PORT` — port to serve on (default: `8501`)

Tests: `python result_viewer_app/test_rank_viewer.py` (stdlib `unittest`, no fixtures outside a temp dir).

## Decoupling: the data contracts

The viewer imports **nothing** from `src/` or `experiments/` — refactors there cannot break it. It consumes only on-disk files:

1. **Run artifacts** written by `RunLogger` into each run dir: `rewards.csv` (with optional `steps` column), `rank_stats.csv`, `hankel_sweep.csv`, `train_diagnostics.csv`, `config.yaml` (opaque text, used only for display and diffing), `figures/*.png`, `trajectories/*.npz`.
2. **Launcher manifests** — `<exp>/cached/*manifest*.json`, as written by e.g. `experiments/src/run_fhrdqn_seeds.py`:

   ```json
   {"seeds": [44, 66],
    "runs": {"<arm>": {"<seed>": "<run dir relative to the exp dir>"}},
    "overrides": {"<arm>": {"<config param>": "value"}}}
   ```

   Every field is optional, unknown fields are ignored, malformed files are skipped. Each manifest **file** is its own *family* (`fhrdqn_runs_manifest.old-lambda0.5.json` → family `fhrdqn_runs.old-lambda0.5`); families never merge, so an archived family's `baseline` is never averaged with the current one. A run dir the manifest doesn't list (still training, superseded by a retrain, or foreign) is flagged and kept out of the tracked arms' seed averages.

## What it shows

Pick a run from the dropdown — grouped by experiment, labelled `arm · seed N · date` for manifest-tracked runs. For each run, the viewer shows:

- **Variant tile** — the run's manifest arm, seed and the config overrides it actually trained with (the `config.yaml` copy does not reflect overrides).
- **Spectra over training** — a slider with play and step controls lets you move through `figures/epNNNNNN_<matrix>.png` one episode at a time; click a spectrum to open it full-screen (with a download button).
- **Persistence across sub-trajectory lengths** — for a selected episode, the chosen metric of the Hankel built from only the first τ steps; follows the scrubber.
- **Rank metrics over training** — small-multiple charts for every scalar metric in `rank_stats.csv`, one line per matrix, synced hover, legend toggling, scrubber marker.
- **Learning curve** from `rewards.csv`, on an episodes or env-steps (samples) axis, as return (with rolling smoothing) or cumulative reward. Older runs of ±1-reward-per-step envs derive steps from `|reward|` and the chart says so.

## Compare mode

The **⇄ Compare** button overlays experiment variants. Grouping is **manifest-first**: tracked runs group by their true arm with correct seeds and show the config diff each arm trained with; runs without a manifest fall back to run-id grouping (seed + timestamp stripped). The list is organised per experiment (most recently active first) with **baseline first, then `exp<N>` in descending order**, a select-all per experiment, seed lists, and update dates. Selection persists across reloads.

- **Configuration differences** card — a table of every config parameter that differs between the selected variants (config copies + manifest overrides), with the differing values highlighted.
- **x-axis** — episodes, or env steps (samples); metric and diagnostics charts re-base onto the same axis.
- **seed view** — `mean ± range`, `overlaid` (colour = variant, dash = seed), or `side-by-side` (identical axes, synced hover).
- **reward y** — episode return (rolling smoothing) or cumulative reward.
- Rank metrics, Hankel sweep evolution and training diagnostics are seed-averaged per variant in the same view. Compare auto-refreshes while runs are still training.

## Publication export & renaming

- Every chart has hover buttons: **⬇ SVG** (vector) and **⬇ PNG** (3× resolution). Exports are standalone figures — white background regardless of theme, title + legend header, labelled x *and* y axes, all styles inlined — ready to drop into a thesis.
- **Double-click any label** — chart titles, legend entries, variant names in compare — to rename it. Renames persist in the browser (localStorage), apply everywhere the label is drawn, and flow into exports. Clear the text (or retype the default) to reset.
