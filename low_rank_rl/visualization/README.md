# `low_rank_rl.visualization`

Pure plotting layer. Every function **takes pre-computed analysis objects
(dataclasses or numpy arrays) and returns a `matplotlib.figure.Figure`**. No
training, no inference, no I/O beyond the optional `save_path` argument.

## Module map

| Module | Plots |
|---|---|
| `training.py`      | `plot_episode_durations`, `plot_learning_curves` |
| `rank_analysis.py` | `plot_singular_value_spectrum`, `plot_hosvd_spectra`, `plot_rank_vs_episode`, `plot_hankel_spectrum`, `plot_shift_comparison` |
| `value_fn.py`      | `plot_value_heatmap`, `plot_q_heatmap` |

## Training plots

### `plot_episode_durations(durations, window=100)`

Raw per-episode duration in light blue plus a `window`-episode rolling mean
overlay. The rolling mean is a simple convolution with a uniform kernel of
size `window`; the overlay is only drawn when the series is at least as long
as the window.

### `plot_learning_curves({label: rewards}, window=50)`

Takes a dict `{run_label: list_of_per_episode_rewards}` and overlays a raw
curve per label (alpha 0.25) plus a bolder rolling mean line per run on the
same axes.

## Rank / spectral plots

### `plot_singular_value_spectrum(metrics: RankMetrics)`

Log-scale plot of $\sigma_1, \sigma_2, \dots$ with two reference lines:

- horizontal dashed line at $10^{-5} \sigma_1$ (the default numerical-rank
  threshold);
- vertical dotted line at `numerical_rank - 0.5`.

Stable rank, effective rank, and spectral gap are annotated in the corner.

### `plot_hosvd_spectra(spectra: dict[int, np.ndarray])`

One sub-plot per mode of the value tensor (see
`analysis.tensor.hosvd_spectra`). Each sub-plot shows that mode's singular
values and annotates its stable rank
$\sum_i \sigma_i^2 / \sigma_1^2$.

Use e.g. `dim_labels=["cos θ₁", "sin θ₁", "cos θ₂", "sin θ₂", "ω₁", "ω₂"]`
(the full 6-D Acrobot obs) or `["x", "v"]` (MountainCar) to label modes
meaningfully.

### `plot_rank_vs_episode(history)`

Takes a list of checkpoint dicts with keys
`{"episode", "stable_rank", "effective_rank", "normalised_rank"}`. Draws one
subplot per metric showing its evolution during training. Useful for
observing rank collapse (or lack thereof) during learning.

### `plot_hankel_spectrum(metrics: HankelMetrics)`

Single-axis log plot of $\sigma_i(H)$ for a Hankel matrix, with the
numerical rank cut-off as a vertical dotted line and stable/effective rank
annotations.

### `plot_shift_comparison(comparison: SuccessorComparison)`

Two-panel side-by-side singular-value spectra for the vanilla $\hat M$ vs
shifted $\tilde M$ successor matrices. Used to visualise the central claim
of arXiv:2509.05193 — the shifted matrix should be markedly lower rank.

## Value-function heatmaps

Both heatmap helpers share a private `_grid_states(env, dims, n_bins, fixed_values)` that constructs an `n_bins × n_bins` grid over the two chosen state
dimensions, leaving all other dimensions pinned to their mid-range (or user-
supplied `fixed_values`).

### `plot_value_heatmap(agent, env, dims=(0, 1), n_bins=50)`

Plots $V(s) = \max_a Q(s, a)$ as a 2-D heatmap over the chosen state slice
using the `viridis` colormap.

### `plot_q_heatmap(agent, env, action, dims=(0, 1), n_bins=50)`

Same slicing convention, but plots $Q(s, a)$ for a fixed action index $a$
using the `plasma` colormap.

For high-dimensional state spaces (e.g. Acrobot is 6-D) a single heatmap is
one 2-D slice through the value landscape; pass `fixed_values={i: v, …}` to
control where the other dimensions are evaluated.

## Backend notes

- All functions return the `Figure`; the caller owns closing. When used in
  scripts, `fig.savefig(path, dpi=150)` is called internally only if
  `save_path` is supplied.
- Tests run with `matplotlib.use("Agg")` so no display is required.

## Tests

`tests/test_visualization.py` verifies each function:

- returns a `matplotlib.figure.Figure`;
- handles both log-scale and linear-scale options;
- accepts both single-mode and multi-mode HOSVD spectra;
- renders value / Q heatmaps for various action indices without error.
