"""Track how predictable a policy's value sequence is, throughout training.

At a checkpoint during training this probe freezes the policy, rolls a set of
greedy trajectories, and asks one question of them: can a single linear
recurrence of order r reproduce the value signal along a trajectory it has
never seen?

For each order in `orders` it fits one GLOBAL recurrence (all training
trajectories pooled) and reports how well that recurrence predicts, under two
independent splits and two prediction regimes.

The two splits
--------------
held_out_trajectory_split
    Fit on the first `training_fraction` of the trajectories, evaluate on the
    remaining ones. Both sets come from the SAME frozen policy, so this asks
    "does one recurrence describe this policy's value sequences in general?"
    and nothing about how the policy changes.

prefix_suffix_split
    Fit on the first half of every trajectory, evaluate on the second half of
    those same trajectories. This asks the forecasting question directly: given
    how a trajectory started, does the recurrence say how it ends?

The two prediction regimes
--------------------------
one_step_ahead
    Each prediction is made from the TRUE previous values, so errors never
    accumulate. Measures raw fit quality, and is the optimistic number.

free_running
    Seeded with `order` true values, then the recurrence runs on its own
    output. Errors compound, so this is the honest test of the low-rank claim:
    a recurrence can look excellent one step ahead and still diverge here.
    Divergence is recorded rather than hidden -- see `free_running_diverged`.

Every number the probe reports is available both as a plain RMSE (in the value
signal's own units) and as a normalised RMSE (divided by the signal's
root-mean-square), the latter being what makes Acrobot and CartPole comparable.
"""
import numpy as np
import torch

from analysis.low_rank.recurrence import (fit_autoregressive_coefficients,
                                          forecast_free_running,
                                          forecast_rolling_horizon,
                                          normalised_root_mean_squared_error,
                                          predict_one_step_ahead,
                                          root_mean_squared_error)

# Orders probed by default. 2 is the order the low-rank work has focused on; 8
# is included so over-parameterisation shows up as free-running divergence.
DEFAULT_ORDERS = (2, 3, 4, 8)

# Forecast window sizes for the rolling-horizon regime: predict this many steps
# from the true history, then observe what really happened and forecast again.
# Spanning powers of two from 1 turns "how predictable is the value signal?"
# into a curve rather than a single number -- horizon 1 is one-step-ahead and a
# horizon past the trajectory length is full free running, so the sweep shows
# exactly how far ahead the recurrence stays useful.
DEFAULT_FORECAST_HORIZONS = (1, 2, 4, 8, 16, 32, 64)

HELD_OUT_TRAJECTORY_SPLIT = "held_out_trajectory_split"
PREFIX_SUFFIX_SPLIT = "prefix_suffix_split"

# A free-running forecast on an unstable recurrence overflows fast. Anything
# beyond this multiple of the true signal's scale is called divergence rather
# than being reported as a meaningless finite error.
DIVERGENCE_SCALE_MULTIPLE = 1e6


def collect_greedy_value_sequences(agent, env, seeds, max_steps=10_000):
    """Roll the frozen greedy policy once per seed, recording the state value.

    The value recorded at each step is V(state) = max over actions of
    Q(state, action), or the dedicated value stream when the network is a
    dueling architecture -- the same convention hankel_policy.py uses, so the
    two analyses describe the same signal.

    Nothing on the agent is mutated: the policy is only read, actions are taken
    by argmax rather than through the agent's exploration policy, and the
    environment passed in should be a probe environment separate from the one
    training is stepping.

    Args:
        agent: a QAgent-like object exposing policy_net and device.
        env: environment to roll in (separate from the training environment).
        seeds: iterable of integer reset seeds, one trajectory per seed.
        max_steps: safety cap for environments that may not terminate.

    Returns:
        List of one-dimensional float arrays, one per seed, each holding the
        value at every step of that trajectory.
    """
    network_has_dueling_streams = hasattr(agent.policy_net, "value_advantage")
    value_sequences = []
    for seed in seeds:
        values = []
        state, _ = env.reset(seed=int(seed))
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action_values = agent.policy_net(state_tensor)[0]
                if network_has_dueling_streams:
                    state_value, _ = agent.policy_net.value_advantage(
                        state_tensor)
                    values.append(float(state_value[0, 0]))
                else:
                    values.append(float(action_values.max()))
            action = int(action_values.argmax())
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
        value_sequences.append(np.asarray(values, dtype=np.float64))
    return value_sequences


def _prediction_metrics(coefficients, value_sequences, order, fit_intercept,
                        free_running_seeds=None):
    """One-step-ahead and free-running errors for a set of sequences.

    Args:
        coefficients: the fitted recurrence.
        value_sequences: sequences to score.
        order: recurrence order (how many values seed the free run).
        free_running_seeds: optional list, one per sequence, of the history the
            free run should start from. Used by the prefix/suffix split, where
            the forecast must start at the end of the prefix rather than at the
            start of the sequence. When None each sequence seeds from its own
            first `order` values and the whole remainder is forecast.

    Returns:
        Dict of metrics, with NaN where a split had nothing long enough to
        score and a boolean recording whether the free run diverged.
    """
    one_step_predictions, one_step_actuals = [], []
    free_running_predictions, free_running_actuals = [], []

    for index, value_sequence in enumerate(value_sequences):
        if len(value_sequence) <= order:
            continue
        one_step_predictions.append(
            predict_one_step_ahead(coefficients, value_sequence,
                                   fit_intercept=fit_intercept))
        one_step_actuals.append(value_sequence[order:])

        if free_running_seeds is None:
            seed_values, actual_future = (value_sequence[:order],
                                          value_sequence[order:])
        else:
            seed_values, actual_future = free_running_seeds[index], value_sequence
        if len(seed_values) < order or len(actual_future) == 0:
            continue
        free_running_predictions.append(
            forecast_free_running(coefficients, seed_values,
                                  horizon=len(actual_future),
                                  fit_intercept=fit_intercept))
        free_running_actuals.append(actual_future)

    def _score(predictions, actuals, label):
        if not predictions:
            return {f"rmse_{label}": float("nan"),
                    f"normalised_rmse_{label}": float("nan")}
        predicted = np.concatenate(predictions)
        actual = np.concatenate(actuals)
        if not np.isfinite(predicted).all():
            return {f"rmse_{label}": float("inf"),
                    f"normalised_rmse_{label}": float("inf")}
        return {f"rmse_{label}": root_mean_squared_error(predicted, actual),
                f"normalised_rmse_{label}":
                    normalised_root_mean_squared_error(predicted, actual)}

    metrics = {}
    metrics.update(_score(one_step_predictions, one_step_actuals,
                          "one_step_ahead"))
    metrics.update(_score(free_running_predictions, free_running_actuals,
                          "free_running"))
    normalised_free_running = metrics["normalised_rmse_free_running"]
    metrics["free_running_diverged"] = bool(
        not np.isfinite(normalised_free_running)
        or normalised_free_running > DIVERGENCE_SCALE_MULTIPLE)
    metrics["n_sequences_scored"] = len(one_step_predictions)
    return metrics


def _rolling_horizon_metrics(coefficients, value_sequences, order,
                             fit_intercept, horizons):
    """Error of the repeated `horizon`-step forecast, for each horizon.

    At every horizon the recurrence is re-seeded from the true values, forecasts
    that many steps, and is then re-anchored on what actually happened -- the
    way such a model would really be used. Horizon 1 reproduces one-step-ahead
    and a horizon longer than the trajectory reproduces free running, so this
    single sweep contains both of the other regimes as its endpoints.
    """
    per_horizon = {}
    for horizon in horizons:
        predicted_blocks, actual_blocks = [], []
        for value_sequence in value_sequences:
            if len(value_sequence) <= order:
                continue
            predicted_blocks.append(forecast_rolling_horizon(
                coefficients, value_sequence, horizon,
                fit_intercept=fit_intercept))
            actual_blocks.append(value_sequence[order:])
        if not predicted_blocks:
            per_horizon[horizon] = {"rmse": float("nan"),
                                    "normalised_rmse": float("nan"),
                                    "diverged": False}
            continue
        predicted = np.concatenate(predicted_blocks)
        actual = np.concatenate(actual_blocks)
        if not np.isfinite(predicted).all():
            per_horizon[horizon] = {"rmse": float("inf"),
                                    "normalised_rmse": float("inf"),
                                    "diverged": True}
            continue
        normalised = normalised_root_mean_squared_error(predicted, actual)
        per_horizon[horizon] = {
            "rmse": root_mean_squared_error(predicted, actual),
            "normalised_rmse": normalised,
            "diverged": bool(normalised > DIVERGENCE_SCALE_MULTIPLE),
        }
    return per_horizon


def _evaluate_held_out_trajectory_split(value_sequences, order,
                                        training_fraction, ridge_penalty,
                                        fit_intercept, horizons):
    """Fit on some trajectories, score on the trajectories left out."""
    n_training = max(1, int(round(training_fraction * len(value_sequences))))
    n_training = min(n_training, len(value_sequences) - 1)
    training_sequences = value_sequences[:n_training]
    test_sequences = value_sequences[n_training:]
    coefficients = fit_autoregressive_coefficients(
        training_sequences, order, ridge_penalty=ridge_penalty,
        fit_intercept=fit_intercept)
    return {
        "coefficients": coefficients,
        "training": _prediction_metrics(coefficients, training_sequences, order,
                                        fit_intercept),
        "test": _prediction_metrics(coefficients, test_sequences, order,
                                    fit_intercept),
        "training_horizons": _rolling_horizon_metrics(
            coefficients, training_sequences, order, fit_intercept, horizons),
        "test_horizons": _rolling_horizon_metrics(
            coefficients, test_sequences, order, fit_intercept, horizons),
        "training_sequences": training_sequences,
        "test_sequences": test_sequences,
    }


def _evaluate_prefix_suffix_split(value_sequences, order, ridge_penalty,
                                  fit_intercept, horizons):
    """Fit on the first half of each trajectory, forecast the second half."""
    prefixes, suffixes, suffix_seed_histories = [], [], []
    for value_sequence in value_sequences:
        midpoint = len(value_sequence) // 2
        if midpoint <= order or len(value_sequence) - midpoint == 0:
            continue
        prefixes.append(value_sequence[:midpoint])
        suffixes.append(value_sequence[midpoint:])
        # The forecast must start from the real values immediately before the
        # suffix, not from the start of the trajectory.
        suffix_seed_histories.append(value_sequence[midpoint - order:midpoint])
    if not prefixes:
        return None
    coefficients = fit_autoregressive_coefficients(
        prefixes, order, ridge_penalty=ridge_penalty,
        fit_intercept=fit_intercept)
    return {
        "coefficients": coefficients,
        "training": _prediction_metrics(coefficients, prefixes, order,
                                        fit_intercept),
        "test": _prediction_metrics(coefficients, suffixes, order,
                                    fit_intercept,
                                    free_running_seeds=suffix_seed_histories),
        "training_horizons": _rolling_horizon_metrics(
            coefficients, prefixes, order, fit_intercept, horizons),
        "test_horizons": _rolling_horizon_metrics(
            coefficients, suffixes, order, fit_intercept, horizons),
        "training_sequences": prefixes,
        "test_sequences": suffixes,
        "suffix_seed_histories": suffix_seed_histories,
    }


def evaluate_autoregressive_orders(value_sequences, orders=DEFAULT_ORDERS,
                                   training_fraction=0.6, ridge_penalty=1e-8,
                                   fit_intercept=True,
                                   forecast_horizons=DEFAULT_FORECAST_HORIZONS):
    """Fit and score every requested order under both splits.

    fit_intercept defaults to True here (unlike the raw primitive) because real
    value sequences sit at a non-zero level, and a zero-mean recurrence would
    otherwise have to spend its coefficients representing that offset.

    Returns:
        Dict keyed by order, each value a dict keyed by split name. Orders whose
        sequences were all too short are omitted. Returns an empty dict when no
        sequence is usable at all.
    """
    usable_sequences = [np.asarray(s, dtype=np.float64) for s in value_sequences
                        if len(s) >= 2]
    results = {}
    for order in orders:
        long_enough = [s for s in usable_sequences if len(s) > order + 1]
        if len(long_enough) < 2:
            continue  # cannot both fit and hold out
        per_split = {
            HELD_OUT_TRAJECTORY_SPLIT: _evaluate_held_out_trajectory_split(
                long_enough, order, training_fraction, ridge_penalty,
                fit_intercept, forecast_horizons),
        }
        prefix_suffix = _evaluate_prefix_suffix_split(
            long_enough, order, ridge_penalty, fit_intercept, forecast_horizons)
        if prefix_suffix is not None:
            per_split[PREFIX_SUFFIX_SPLIT] = prefix_suffix
        results[order] = per_split
    return results


def metric_rows(episode, results, value_sequences):
    """Flatten `results` into one row per (order, split, subset) for the CSV."""
    mean_length = (float(np.mean([len(s) for s in value_sequences]))
                   if value_sequences else float("nan"))
    rows = []
    for order, per_split in sorted(results.items()):
        for split_name, split in per_split.items():
            for subset in ("training", "test"):
                metrics = split[subset]
                rows.append({
                    "episode": episode,
                    "order": order,
                    "split": split_name,
                    "subset": subset,
                    "rmse_one_step_ahead": metrics["rmse_one_step_ahead"],
                    "normalised_rmse_one_step_ahead":
                        metrics["normalised_rmse_one_step_ahead"],
                    "rmse_free_running": metrics["rmse_free_running"],
                    "normalised_rmse_free_running":
                        metrics["normalised_rmse_free_running"],
                    "free_running_diverged":
                        int(metrics["free_running_diverged"]),
                    "n_sequences_scored": metrics["n_sequences_scored"],
                    "n_trajectories_collected": len(value_sequences),
                    "mean_trajectory_length": mean_length,
                })
    return rows


def horizon_metric_rows(episode, results, value_sequences):
    """Flatten the rolling-horizon sweep into one row per
    (order, split, subset, forecast horizon)."""
    mean_length = (float(np.mean([len(s) for s in value_sequences]))
                   if value_sequences else float("nan"))
    rows = []
    for order, per_split in sorted(results.items()):
        for split_name, split in per_split.items():
            for subset in ("training", "test"):
                for horizon, metrics in sorted(
                        split[f"{subset}_horizons"].items()):
                    rows.append({
                        "episode": episode,
                        "order": order,
                        "split": split_name,
                        "subset": subset,
                        "forecast_horizon": horizon,
                        "rmse": metrics["rmse"],
                        "normalised_rmse": metrics["normalised_rmse"],
                        "diverged": int(metrics["diverged"]),
                        "mean_trajectory_length": mean_length,
                    })
    return rows


def coefficient_rows(episode, results, fit_intercept=True):
    """Flatten fitted coefficients into one row per (order, split, lag).

    `lag` is 1-based and counts back from the present, so lag 1 is the weight on
    the immediately preceding value. The intercept, when fitted, is written with
    lag 0 so a plot can filter it out or show it separately.
    """
    rows = []
    for order, per_split in sorted(results.items()):
        for split_name, split in per_split.items():
            coefficients = split["coefficients"]
            lag_weights = coefficients[:-1] if fit_intercept else coefficients
            for lag_index, coefficient in enumerate(lag_weights, start=1):
                rows.append({"episode": episode, "order": order,
                             "split": split_name, "lag": lag_index,
                             "coefficient": float(coefficient)})
            if fit_intercept:
                rows.append({"episode": episode, "order": order,
                             "split": split_name, "lag": 0,
                             "coefficient": float(coefficients[-1])})
    return rows


def example_rollout_arrays(results, n_examples=2, fit_intercept=True,
                           forecast_horizons=DEFAULT_FORECAST_HORIZONS):
    """Actual vs predicted sequences for a few trajectories, for plotting.

    Picks the first `n_examples` trajectories of each subset of each split, and
    for each stores the true values alongside both the one-step-ahead and the
    free-running prediction. Keys are flat strings so the whole thing saves as
    one .npz that the notebook and the viewer can both read:

        order2__held_out_trajectory_split__test__0__actual
        order2__held_out_trajectory_split__test__0__one_step_ahead
        order2__held_out_trajectory_split__test__0__free_running
    """
    arrays = {}
    for order, per_split in sorted(results.items()):
        for split_name, split in per_split.items():
            coefficients = split["coefficients"]
            for subset in ("training", "test"):
                sequences = split[f"{subset}_sequences"]
                for index, value_sequence in enumerate(sequences[:n_examples]):
                    if len(value_sequence) <= order:
                        continue
                    prefix = f"order{order}__{split_name}__{subset}__{index}"
                    arrays[f"{prefix}__actual"] = value_sequence
                    arrays[f"{prefix}__one_step_ahead"] = predict_one_step_ahead(
                        coefficients, value_sequence,
                        fit_intercept=fit_intercept)
                    seeds = split.get("suffix_seed_histories")
                    seed_values = (seeds[index]
                                   if seeds is not None and subset == "test"
                                   else value_sequence[:order])
                    horizon = (len(value_sequence)
                               if seeds is not None and subset == "test"
                               else len(value_sequence) - order)
                    arrays[f"{prefix}__free_running"] = forecast_free_running(
                        coefficients, seed_values, horizon=horizon,
                        fit_intercept=fit_intercept)
                    # The rolling-horizon forecast at each configured window, so
                    # the plots can show the re-anchoring sawtooth rather than
                    # only the seeded-once extreme. Aligned with
                    # value_sequence[order:], like one_step_ahead.
                    for forecast_horizon in forecast_horizons:
                        arrays[f"{prefix}__rolling_horizon_{forecast_horizon}"] = \
                            forecast_rolling_horizon(
                                coefficients, value_sequence, forecast_horizon,
                                fit_intercept=fit_intercept)
    return arrays


def autoregressive_value_probe(agent, env, orders=DEFAULT_ORDERS,
                               n_trajectories=12, base_seed=40_000,
                               training_fraction=0.6, ridge_penalty=1e-8,
                               fit_intercept=True, n_example_rollouts=2,
                               max_steps=10_000,
                               forecast_horizons=DEFAULT_FORECAST_HORIZONS,
                               episode=None, run_logger=None):
    """Entry point called from the analysis registry at every checkpoint.

    Collects greedy trajectories, evaluates every order under both splits, and
    -- when a run_logger is supplied -- appends the metric and coefficient rows
    and saves the example rollouts for later plotting.

    Returns the raw results dict so a notebook can call this directly and plot
    without going through the logger.
    """
    seeds = [base_seed + i for i in range(n_trajectories)]
    value_sequences = collect_greedy_value_sequences(
        agent, env, seeds, max_steps=max_steps)
    results = evaluate_autoregressive_orders(
        value_sequences, orders=orders, training_fraction=training_fraction,
        ridge_penalty=ridge_penalty, fit_intercept=fit_intercept,
        forecast_horizons=tuple(forecast_horizons))
    if run_logger is not None and results:
        run_logger.log_autoregressive_metrics(
            metric_rows(episode, results, value_sequences))
        run_logger.log_autoregressive_horizon_metrics(
            horizon_metric_rows(episode, results, value_sequences))
        run_logger.log_autoregressive_coefficients(
            coefficient_rows(episode, results, fit_intercept=fit_intercept))
        run_logger.save_autoregressive_example_rollouts(
            episode, example_rollout_arrays(
                results, n_examples=n_example_rollouts,
                fit_intercept=fit_intercept,
                forecast_horizons=tuple(forecast_horizons)))
    return {"value_sequences": value_sequences, "results": results}
