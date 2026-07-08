# Result viewer app

A local, dependency-free web viewer for [`RunLogger`](../src/analysis/run_logger.py)
outputs. It scans `<exp>/runs/<run_id>/` directories and gives you a logical way to
watch how the **low-rank structure** of the Hankel / Q matrices evolves over
training — stepping through the spectra one episode at a time and plotting every
rank metric across the whole run.

It reads the logger's outputs **read-only** and lives entirely outside `src/`; it
does not touch training code.

## Run it

```bash
python result_viewer_app/rank_viewer.py            # http://localhost:8501, root = ./experiments
python result_viewer_app/rank_viewer.py --root /scratch/experiments --port 9000
```

Then open the printed URL in a browser. Pure Python standard library — no
`pip install`, no build step — so you can also `scp` `rank_viewer.py` +
`rank_viewer.html` to the HPC and point `--root` at the runs directory there.

Options:

- `--root PATH` — directory containing `<exp>/runs/<run_id>/` trees (default `./experiments`).
- `--host HOST` — bind address (default `127.0.0.1`).
- `--port PORT` — default `8501`.

## What it shows

Pick a run from the dropdown (newest first). For that run you get:

- **Spectra over training** — a slider with play / step (‹ ▶ ›) controls that walks
  through the `figures/epNNNNNN_<matrix>.png` spectra episode by episode, one panel
  per matrix (Hankel Q / V / A), so you can literally watch the singular-value tail
  collapse. Press play to animate the whole run, or drag to any episode. **Click any
  spectrum to enlarge it full-screen** (Esc or click the backdrop to close).
- **Persistence across sub-trajectory lengths** — for a single episode, the chosen
  metric of the Hankel built from only the first τ steps, one line per matrix (mean
  across rollouts, min–max band). Pick the **metric** and the **episode** from the
  dropdowns in the card header — or just scrub the spectra above and this chart
  follows the episode you're looking at.
- **Rank metrics over training** — small-multiple line charts of every scalar metric
  in `rank_stats.csv`: effective rank, stable rank, spikiness, row / column coherence,
  peak row / column score (irs / ics), and non-zero rows / columns — one line per
  matrix. Toggle matrices in the legend (applies to every chart), hover for a synced
  crosshair readout, and note the **dashed vertical marker** on each chart — it tracks
  the episode selected in the scrubber above, so the spectra and the curves stay in step.
- **Episode reward** from `rewards.csv`, plus headline stat tiles (best / final
  reward, matrices tracked, analysis ticks).
- The frozen `config.yaml` for the run (collapsible).

Light / dark theme follows your OS and can be toggled with the ◐ button. Series
colors use a colorblind-safe categorical palette and every series is also
directly labeled, so identity never depends on color alone.

## Live mode — works while training is still running

The viewer is built to be left open **while the notebook / training loop is still
writing**. The **Live** toggle in the header (on by default) polls the run every
~4 seconds and folds in whatever is new — appended `rank_stats.csv` rows, a
rewritten `rewards.csv`, and freshly written spectrum PNGs — without a page reload:

- The dot pulses green while polling; the timestamp shows the last successful
  refresh. If the server goes away it shows **Reconnecting** and keeps trying.
- Your place is preserved across refreshes: the scrubber episode, the hidden
  matrices, and the scroll position all stay put. If you've parked the scrubber on
  the **last** frame it auto-advances to each new episode as it lands (tailing the
  live front); otherwise it leaves your chosen episode alone.
- Refreshes never interrupt you mid-gesture — a poll that arrives while you're
  dragging the slider or playing the animation is held and applied the moment you
  stop.
- New runs that appear on disk show up in the dropdown automatically; use **⟳** to
  force an immediate refresh, or untick **Live** to freeze the view.

The server re-reads the run from disk on every request and tolerates a half-written
CSV (a torn final line during a `rewards.csv` rewrite, or a ragged `rank_stats.csv`
row) by skipping the bad line rather than erroring — so polling a run that's
actively being written is safe.

## Files

- `rank_viewer.py` — stdlib HTTP server: run discovery, CSV / figure parsing (robust
  to partial writes), static serving, and a path-traversal guard on the figure route.
- `rank_viewer.html` — the single-page frontend (inline CSS / JS, no external deps):
  charts, the spectrum scrubber, and the live-polling loop.
