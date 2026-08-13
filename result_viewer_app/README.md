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

## What it shows

Pick a run from the dropdown, with the newest runs shown first. For each run, the viewer shows:

- **Spectra over training** — a slider with play and step controls lets you move through `figures/epNNNNNN_<matrix>.png` one episode at a time. Each matrix gets its own panel, so you can watch the singular-value tail change over training. You can play the full sequence, scrub to any episode, and click a spectrum to open it full-screen.
- **Persistence across sub-trajectory lengths** — for a selected episode, the app plots the chosen metric of the Hankel matrix built from only the first \( \tau \) steps. Each matrix is shown as a separate line, with the mean across rollouts and a min-max band. You can select both the metric and the episode directly, or just scrub the spectra above and let this chart follow along.
- **Rank metrics over training** — small-multiple charts show every scalar metric in `rank_stats.csv`, including effective rank, stable rank, spikiness, row and column coherence, peak row and column score (`irs`, `ics`), and non-zero rows and columns. Each chart shows one line per matrix, supports synced hover, lets you toggle matrices in the legend across all charts, and includes a dashed vertical marker tied to the selected episode in the scrubber.
- **Learning curve** from `rewards.csv`, with summary tiles (best/final reward, total env steps). The x-axis switches between **episodes** and **environment steps (samples)** — the axis learning curves are reported on in the literature — and the y-axis between per-episode return (with optional rolling smoothing) and **cumulative reward**. New runs log a per-episode `steps` column; for older runs of CartPole / MountainCar / Acrobot the steps are derived from `|reward|` (those envs pay ±1 per step) and the chart says so.

## Compare mode

The **⇄ Compare** button overlays several experiment variants — e.g. FHR-DQN vs HR-DQN vs baseline DQN — with runs grouped by run id after stripping the seed and launch timestamp. Controls at the top pick:

- **x-axis** — episodes, or env steps (samples) so variants with different episode lengths stay comparable. Metric and diagnostics charts are re-based onto the same axis.
- **seed view** — `mean ± range` (one line per variant: seed mean with a min–max band; on the steps axis seeds are aligned by interpolation onto a shared grid), `overlaid` (every seed its own line — colour = variant, dash = seed), or `side-by-side` (one panel per variant with identical axes and hover synced across panels, so seeds line up point-for-point).
- **reward y** — episode return (with rolling smoothing) or cumulative reward, where a gap between variants on the steps axis reads directly as a sample-efficiency gap.

Rank metrics (`rank_stats.csv`), the Hankel sweep evolution, and training diagnostics (`train_diagnostics.csv`) are seed-averaged per variant in the same view.
