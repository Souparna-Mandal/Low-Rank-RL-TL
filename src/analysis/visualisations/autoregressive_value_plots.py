"""Plots for the autoregressive value-recurrence probe.

Reads the artifacts a training run wrote under runs/<run id>/ --
autoregressive_value_metrics.csv, autoregressive_value_coefficients.csv and
autoregressive_rollouts/epNNNNNN.npz -- and renders the four things the
experiment is about:

    plot_prediction_error_over_training  how predictable the value signal is,
                                         and how that changes as the agent learns
    plot_coefficient_evolution           one figure per recurrence order, showing
                                         every coefficient's path over training
    plot_example_rollouts                actual vs predicted value along single
                                         trajectories, from training and test sets
    plot_reward_with_checkpoints         the learning curve, with probe
                                         checkpoints marked, for context

Every function returns its matplotlib Figure so a notebook can adjust or save
it, and takes save_to to write a PNG directly.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HELD_OUT_TRAJECTORY_SPLIT = "held_out_trajectory_split"
PREFIX_SUFFIX_SPLIT = "prefix_suffix_split"

SPLIT_TITLES = {
    HELD_OUT_TRAJECTORY_SPLIT: "held-out trajectories",
    PREFIX_SUFFIX_SPLIT: "prefix / suffix within trajectory",
}
SPLIT_DESCRIPTIONS = {
    HELD_OUT_TRAJECTORY_SPLIT:
        "fitted on some trajectories, scored on trajectories from the same "
        "frozen policy that it never saw",
    PREFIX_SUFFIX_SPLIT:
        "fitted on the first half of every trajectory, scored on forecasting "
        "the second half",
}
def load_prediction_errors(run_directory):
    """The per-checkpoint one-step-ahead fit quality as a DataFrame.

    Errors at longer forecast windows live in the horizon sweep -- see
    load_horizon_errors.
    """
    path = pathlib.Path(run_directory) / "autoregressive_value_metrics.csv"
    return pd.read_csv(path)


def load_coefficients(run_directory):
    """The per-checkpoint fitted coefficients as a DataFrame."""
    path = pathlib.Path(run_directory) / "autoregressive_value_coefficients.csv"
    return pd.read_csv(path)


def load_example_rollouts(run_directory, episode=None):
    """One checkpoint's example rollouts as a plain dict of arrays.

    episode=None loads the last checkpoint available.
    """
    rollout_directory = pathlib.Path(run_directory) / "autoregressive_rollouts"
    files = sorted(rollout_directory.glob("ep*.npz"))
    if not files:
        raise FileNotFoundError(f"no rollout files under {rollout_directory}")
    if episode is None:
        chosen = files[-1]
    else:
        chosen = rollout_directory / f"ep{episode:06d}.npz"
        if not chosen.exists():
            raise FileNotFoundError(
                f"no rollouts for episode {episode}; available: "
                f"{[int(f.stem[2:]) for f in files]}")
    with np.load(chosen) as data:
        return {key: data[key] for key in data.files}, int(chosen.stem[2:])


def load_horizon_errors(run_directory):
    """The rolling-horizon forecast sweep as a DataFrame.

    One row per (episode, order, split, subset, forecast_horizon), where the
    horizon is how many steps are predicted before the recurrence is re-anchored
    on the values that actually occurred.
    """
    path = (pathlib.Path(run_directory)
            / "autoregressive_value_horizon_metrics.csv")
    return pd.read_csv(path)


def plot_error_versus_forecast_horizon(
        run_directory, episode=None, split=HELD_OUT_TRAJECTORY_SPLIT,
        subset="test", save_to=None, show=True, figsize=(8, 5)):
    """Error against how far ahead the recurrence is asked to forecast.

    At horizon tau the recurrence predicts tau steps from the true history, is
    then re-anchored on what actually happened, and predicts the next tau. The
    two endpoints of this curve are the other regimes: horizon 1 IS
    one-step-ahead, and a horizon past the trajectory length IS free running.

    Read it as: how far ahead can this recurrence see before it stops being
    useful? A curve that rises then flattens has saturated -- the recurrence has
    decayed to predicting the signal's mean, so the error plateaus at roughly
    the signal's own scale rather than growing without bound. A curve that keeps
    climbing steeply is an unstable fit.

    episode=None uses the last checkpoint.
    """
    errors = load_horizon_errors(run_directory)
    errors = errors[(errors["split"] == split) & (errors["subset"] == subset)]
    if episode is None:
        episode = errors["episode"].max()
    at_episode = errors[errors["episode"] == episode]
    if at_episode.empty:
        raise ValueError(f"no horizon rows at episode {episode}")
    orders = sorted(at_episode["order"].unique())
    colours = _order_colours(orders)

    figure, axis = plt.subplots(figsize=figsize)
    for order in orders:
        selected = (at_episode[at_episode["order"] == order]
                    .sort_values("forecast_horizon"))
        axis.plot(selected["forecast_horizon"], selected["one_minus_r_squared"],
                  marker="o", markersize=4, color=colours[order],
                  linewidth=1.8, label=f"order {order}")
    axis.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    axis.annotate("no better than predicting the mean", xy=(0.02, 1.05),
                  xycoords=("axes fraction", "data"), fontsize=8, color="grey")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("forecast horizon $\\tau$ (steps predicted before re-anchoring)")
    axis.set_ylabel("1 - R$^2$  (fraction of variance unexplained)")
    axis.set_title(f"How far ahead can the recurrence see?\n"
                   f"episode {episode} · {SPLIT_TITLES.get(split, split)} · {subset}")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def plot_horizon_error_over_training(
        run_directory, order=2, split=HELD_OUT_TRAJECTORY_SPLIT,
        subset="test", save_to=None, show=True, figsize=(10, 4.5)):
    """For one recurrence order, the error at each horizon across training.

    Shows whether the agent's value signal becomes predictable further ahead as
    it learns, or only ever one step ahead.
    """
    errors = load_horizon_errors(run_directory)
    errors = errors[(errors["split"] == split) & (errors["subset"] == subset)
                    & (errors["order"] == order)]
    if errors.empty:
        raise ValueError(f"no horizon rows for order {order}, split {split!r}")
    horizons = sorted(errors["forecast_horizon"].unique())
    colour_map = plt.get_cmap("plasma")

    figure, axis = plt.subplots(figsize=figsize)
    for index, horizon in enumerate(horizons):
        selected = (errors[errors["forecast_horizon"] == horizon]
                    .sort_values("episode"))
        axis.plot(selected["episode"], selected["one_minus_r_squared"],
                  color=colour_map(index / max(len(horizons) - 1, 1)),
                  linewidth=1.6, label=f"$\\tau$ = {horizon}")
    axis.set_yscale("log")
    axis.set_xlabel("training episode")
    axis.set_ylabel("1 - R$^2$  (fraction of variance unexplained)")
    axis.set_title(f"Order-{order} forecast error at each horizon, over training "
                   f"({SPLIT_TITLES.get(split, split)}, {subset})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def summarise_horizon_sweep(run_directory, episode=None,
                            split=HELD_OUT_TRAJECTORY_SPLIT, subset="test",
                            metric="one_minus_r_squared"):
    """Error at each (order, forecast window) as a readable pivot table.

    metric is "one_minus_r_squared" or "rmse"; call once with each to report
    both.
    """
    errors = load_horizon_errors(run_directory)
    errors = errors[(errors["split"] == split) & (errors["subset"] == subset)]
    if episode is None:
        episode = errors["episode"].max()
    at_episode = errors[errors["episode"] == episode]
    table = at_episode.pivot_table(index="order", columns="forecast_horizon",
                                   values=metric)
    table.columns = [f"tau={int(c)}" for c in table.columns]
    return table


def find_latest_run_with_probe(runs_directory="runs"):
    """The most recently modified run directory that actually has probe output.

    Selecting by name is not safe: an experiment directory can hold runs named
    by timestamp alongside hand-named ones (winning_s0, hrdqn_s0, ...) which
    sort after any digit-led name. This picks by modification time among the
    runs that contain the probe artifacts, so it cannot return a run from a
    different experiment that happens to sort last.
    """
    runs_directory = pathlib.Path(runs_directory)
    candidates = [d for d in runs_directory.glob("*")
                  if d.is_dir()
                  and (d / "autoregressive_value_metrics.csv").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"no run under {runs_directory} has autoregressive probe output; "
            "run the experiment first (see the cell above)")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def available_orders(run_directory, split=HELD_OUT_TRAJECTORY_SPLIT):
    """Recurrence orders this run actually fitted, ascending.

    Read this rather than hardcoding a list: the orders come from the config's
    autoregressive_value_probe.orders, so a run configured with [2, 4] has no
    order 3 or 8 and asking for one is an error, not an empty plot.
    """
    coefficients = load_coefficients(run_directory)
    coefficients = coefficients[coefficients["split"] == split]
    return sorted(int(order) for order in coefficients["order"].unique())


def fitted_with_intercept(run_directory, split=HELD_OUT_TRAJECTORY_SPLIT):
    """Whether this run fitted an intercept, inferred from the artifacts.

    The intercept is written as the lag-0 row, so its absence means the run set
    autoregressive_value_probe.fit_intercept to false.
    """
    coefficients = load_coefficients(run_directory)
    coefficients = coefficients[coefficients["split"] == split]
    return bool((coefficients["lag"] == 0).any())


def available_rollout_episodes(run_directory):
    """Checkpoint episodes for which example rollouts were saved."""
    rollout_directory = pathlib.Path(run_directory) / "autoregressive_rollouts"
    return sorted(int(p.stem[2:]) for p in rollout_directory.glob("ep*.npz"))


def _order_colours(orders):
    colour_map = plt.get_cmap("viridis")
    return {order: colour_map(i / max(len(orders) - 1, 1))
            for i, order in enumerate(sorted(orders))}


def plot_prediction_error_over_training(
        run_directory, split=HELD_OUT_TRAJECTORY_SPLIT,
        unexplained_variance=True, save_to=None, show=True, figsize=None):
    """How well each recurrence order predicts, checkpoint by checkpoint.

    One column per recurrence order, one row per subset (training, test);
    within a panel, one line per forecast window tau. At window tau the
    recurrence predicts tau steps from the true history, is then re-anchored
    on the values that actually occurred, and predicts the next tau -- so
    tau = 1 is the one-step-ahead fit and larger windows show how quickly
    compounding errors lose the signal. The test row is the one that matters:
    those trajectories were never seen during fitting.

    unexplained_variance=True plots 1 - R^2, the fraction of the signal's
    variance left unexplained: 0 is perfect, 1 is no better than predicting the
    signal's own mean, above 1 is worse than that. Set it False to plot the raw
    RMSE instead, which is in the value signal's own units and therefore not
    comparable between environments.
    """
    horizon_errors = load_horizon_errors(run_directory)
    horizon_errors = horizon_errors[horizon_errors["split"] == split]
    if horizon_errors.empty:
        raise ValueError(f"no rows for split {split!r} in {run_directory}")
    metric = "one_minus_r_squared" if unexplained_variance else "rmse"
    orders = sorted(horizon_errors["order"].unique())
    horizons = sorted(horizon_errors["forecast_horizon"].unique())
    colour_map = plt.get_cmap("plasma")
    if figsize is None:
        figsize = (max(11, 3.4 * len(orders)), 7)

    figure, axes = plt.subplots(2, len(orders), figsize=figsize, sharex=True,
                                squeeze=False)
    for column, order in enumerate(orders):
        for row, subset in enumerate(("training", "test")):
            axis = axes[row][column]
            if 1 not in horizons:
                # The one-step-ahead fit is the tau = 1 endpoint of the sweep;
                # draw it from the metrics file when the sweep skipped it.
                one_step = load_prediction_errors(run_directory)
                selected = one_step[(one_step["split"] == split)
                                    & (one_step["order"] == order)
                                    & (one_step["subset"] == subset)]
                selected = selected.sort_values("episode")
                axis.plot(selected["episode"],
                          selected[f"{metric}_one_step_ahead"],
                          color="black", linewidth=1.6,
                          label="one step ahead ($\\tau$ = 1)")
            for index, horizon in enumerate(horizons):
                selected = horizon_errors[
                    (horizon_errors["order"] == order)
                    & (horizon_errors["subset"] == subset)
                    & (horizon_errors["forecast_horizon"] == horizon)]
                selected = selected.sort_values("episode")
                axis.plot(selected["episode"], selected[metric],
                          color=colour_map(index / max(len(horizons) - 1, 1)),
                          linewidth=1.5, label=f"$\\tau$ = {horizon}")
            axis.set_title(f"order {order} — {subset}", fontsize=10)
            axis.set_yscale("log")
            axis.grid(alpha=0.3)
            if row == 1:
                axis.set_xlabel("training episode")
            if unexplained_variance:
                # 1 is the reference: no better than predicting the mean.
                axis.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
            if column == 0:
                axis.set_ylabel("1 - $R^2$  (variance unexplained)"
                                if unexplained_variance else "RMSE")
    axes[0][0].legend(fontsize=8, ncol=2)
    figure.suptitle(
        f"Value predictability over training, by forecast window — "
        f"{SPLIT_TITLES.get(split, split)}\n"
        f"{SPLIT_DESCRIPTIONS.get(split, '')}", fontsize=11)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def plot_coefficient_evolution(run_directory, order,
                               split=HELD_OUT_TRAJECTORY_SPLIT,
                               include_intercept=False, save_to=None,
                               show=True, figsize=(9, 4.5)):
    """Every fitted coefficient of ONE recurrence order, across training.

    One line per lag: lag 1 is the weight on the immediately preceding value.
    The intercept (lag 0) is on a different scale from the lag weights, so it is
    off by default and gets its own twin axis when switched on.

    The dashed line marks the sum of the lag weights. A sum near 1 is the
    persistence / unit-root regime, where the recurrence mostly says "the value
    stays where it is".
    """
    coefficients = load_coefficients(run_directory)
    coefficients = coefficients[(coefficients["order"] == order)
                                & (coefficients["split"] == split)]
    if coefficients.empty:
        raise ValueError(
            f"no coefficients for order {order}, split {split!r}. This run "
            f"fitted orders {available_orders(run_directory, split)} — see "
            f"autoregressive_value_probe.orders in its config.yaml. Use "
            f"available_orders(run_directory) instead of a hardcoded list.")

    lag_weights = coefficients[coefficients["lag"] > 0]
    figure, axis = plt.subplots(figsize=figsize)
    lags = sorted(lag_weights["lag"].unique())
    colour_map = plt.get_cmap("tab10")
    for index, lag in enumerate(lags):
        selected = lag_weights[lag_weights["lag"] == lag].sort_values("episode")
        axis.plot(selected["episode"], selected["coefficient"],
                  label=f"lag {lag}", color=colour_map(index % 10), linewidth=1.8)

    coefficient_sum = (lag_weights.groupby("episode")["coefficient"].sum()
                       .sort_index())
    axis.plot(coefficient_sum.index, coefficient_sum.values, "k--",
              linewidth=1.2, alpha=0.7, label="sum of lag weights")
    axis.axhline(1.0, color="grey", linewidth=0.8, alpha=0.5)
    axis.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)

    if include_intercept:
        intercepts = coefficients[coefficients["lag"] == 0].sort_values("episode")
        if not intercepts.empty:
            intercept_axis = axis.twinx()
            intercept_axis.plot(intercepts["episode"], intercepts["coefficient"],
                                color="tab:red", linestyle=":", linewidth=1.5,
                                label="intercept")
            intercept_axis.set_ylabel("intercept", color="tab:red")
            intercept_axis.tick_params(axis="y", labelcolor="tab:red")

    axis.set_xlabel("training episode")
    axis.set_ylabel("coefficient")
    axis.set_title(f"Order-{order} recurrence coefficients over training "
                   f"({SPLIT_TITLES.get(split, split)})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def plot_example_rollouts(run_directory, episode=None, orders=None,
                          split=HELD_OUT_TRAJECTORY_SPLIT,
                          subsets=("training", "test"), example_index=0,
                          forecast_horizon=None,
                          save_to=None, show=True, figsize=(13, 3.2)):
    """Actual vs predicted value along single trajectories.

    One column per recurrence order, one row per subset, so the training and
    test rows sit directly above one another for the same order. Each rolling
    forecast line predicts tau steps from the true history before being
    re-anchored on what actually happened, so its gap to the actual line is
    how fast the recurrence loses the signal within a window of that size.

    forecast_horizon=None overlays every window the run saved; pass a single
    tau to show just that one alongside the one-step-ahead fit.
    """
    arrays, resolved_episode = load_example_rollouts(run_directory, episode)
    if orders is None:
        orders = sorted({int(key.split("__")[0][5:]) for key in arrays
                         if key.endswith("__actual")})
    colour_map = plt.get_cmap("plasma")
    figure, axes = plt.subplots(len(subsets), len(orders),
                                figsize=(figsize[0], figsize[1] * len(subsets)),
                                squeeze=False)
    for row, subset in enumerate(subsets):
        for column, order in enumerate(orders):
            axis = axes[row][column]
            stem = f"order{order}__{split}__{subset}__{example_index}"
            actual = arrays.get(f"{stem}__actual")
            if actual is None:
                axis.set_visible(False)
                continue
            one_step = arrays.get(f"{stem}__one_step_ahead")
            saved_horizons = sorted(
                int(key.rsplit("_", 1)[1]) for key in arrays
                if key.startswith(f"{stem}__rolling_horizon_"))
            if forecast_horizon is None:
                chosen_horizons = saved_horizons
            elif forecast_horizon in saved_horizons:
                chosen_horizons = [forecast_horizon]
            else:
                raise ValueError(
                    f"this run saved no rolling-horizon arrays for "
                    f"tau={forecast_horizon}; available: "
                    f"{saved_horizons or 'none'}")
            axis.plot(np.arange(len(actual)), actual, color="black",
                      linewidth=2.0, label="actual")
            if one_step is not None:
                axis.plot(np.arange(order, order + len(one_step)), one_step,
                          color="tab:blue", linewidth=1.3, alpha=0.9,
                          label="one step ahead")
            for index, horizon in enumerate(chosen_horizons):
                forecast = arrays[f"{stem}__rolling_horizon_{horizon}"]
                # Rolling-horizon arrays align with actual[order:], like the
                # one-step-ahead ones.
                axis.plot(np.arange(order, order + len(forecast)), forecast,
                          color=colour_map(
                              index / max(len(chosen_horizons) - 1, 1)),
                          linewidth=1.2, linestyle="--", alpha=0.9,
                          label=f"$\\tau$ = {horizon}")
            if chosen_horizons:
                # A long window can diverge; keep the true signal readable.
                span = float(np.max(actual) - np.min(actual)) or 1.0
                axis.set_ylim(float(np.min(actual)) - span,
                              float(np.max(actual)) + span)
            axis.set_title(f"order {order} — {subset}", fontsize=9)
            axis.grid(alpha=0.3)
            if row == len(subsets) - 1:
                axis.set_xlabel("t (steps)")
            if column == 0:
                axis.set_ylabel("value")
    axes[0][0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        f"Example value rollouts at episode {resolved_episode} — "
        f"{SPLIT_TITLES.get(split, split)}", fontsize=11)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def plot_reward_with_checkpoints(run_directory, save_to=None, show=True,
                                 figsize=(11, 3.5), smoothing_window=20):
    """Learning curve with probe checkpoints marked, for context.

    Without this it is hard to tell whether a change in predictability lines up
    with the agent actually learning something.
    """
    rewards = pd.read_csv(pathlib.Path(run_directory) / "rewards.csv")
    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(rewards["episode"], rewards["reward"], color="lightsteelblue",
              linewidth=0.8, label="episode reward")
    if smoothing_window > 1 and len(rewards) >= smoothing_window:
        smoothed = rewards["reward"].rolling(smoothing_window).mean()
        axis.plot(rewards["episode"], smoothed, color="tab:blue", linewidth=2.0,
                  label=f"{smoothing_window}-episode mean")
    for checkpoint in available_rollout_episodes(run_directory):
        axis.axvline(checkpoint, color="grey", alpha=0.18, linewidth=0.8)
    axis.set_xlabel("training episode")
    axis.set_ylabel("reward")
    axis.set_title("Learning curve (vertical lines mark probe checkpoints)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _finish(figure, save_to, show)
    return figure


def summarise_final_checkpoint(run_directory,
                               split=HELD_OUT_TRAJECTORY_SPLIT,
                               metric="one_minus_r_squared"):
    """Held-out test error at the last checkpoint, one row per recurrence
    order and one column per forecast window.

    Each "tau=k" column re-anchors on the true values after every k predicted
    steps, so tau=1 is the one-step-ahead fit; a separate "one step" column is
    added only when the sweep skipped window 1. metric is "one_minus_r_squared"
    or "rmse"; call once with each to report both.
    """
    errors = load_prediction_errors(run_directory)
    errors = errors[(errors["split"] == split) & (errors["subset"] == "test")]
    last_episode = errors["episode"].max()
    table = summarise_horizon_sweep(run_directory, episode=last_episode,
                                    split=split, subset="test", metric=metric)
    if "tau=1" not in table.columns:
        final = errors[errors["episode"] == last_episode].set_index("order")
        table.insert(0, "one step", final[f"{metric}_one_step_ahead"])
    return table


def _finish(figure, save_to, show):
    if save_to is not None:
        pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_to, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
