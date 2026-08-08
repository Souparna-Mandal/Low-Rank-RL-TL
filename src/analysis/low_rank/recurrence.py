"""Autoregressive (linear recurrence) fitting and prediction for value sequences.

The low-Hankel-rank observation says that if the Hankel matrix built from a
scalar sequence has rank at most `order`, then by Kronecker's theorem the
sequence obeys a linear recurrence of that order:

    value[t] = coefficient_1 * value[t-1] + ... + coefficient_order * value[t-order]

These helpers fit those coefficients by least squares and extrapolate with
them, which is the direct test of whether that structural property is of any
predictive use on real value sequences produced by a trained policy.

Naming convention in this module: `value_sequence` is a single one-dimensional
array of scalar values along one trajectory, and `value_sequences` is a list of
such arrays (typically one per trajectory).
"""
import numpy as np


def fit_autoregressive_coefficients(value_sequences, order, ridge_penalty=1e-8,
                                    fit_intercept=False):
    """Least-squares fit of one GLOBAL order-`order` recurrence across sequences.

    Every valid window of every sequence contributes one row, so the returned
    coefficients describe all the supplied trajectories jointly rather than
    being fitted per trajectory.

    Args:
        value_sequences: iterable of one-dimensional arrays of scalar values.
        order: number of past values the recurrence looks back over.
        ridge_penalty: small diagonal added before solving, to keep the normal
            equations well conditioned when the sequences are nearly collinear
            (which they are whenever the true order is lower than `order`).
        fit_intercept: also fit a constant offset term.

    Returns:
        Array of shape (order,), where element 0 is the weight on the MOST
        RECENT value and element order-1 the weight on the oldest. When
        fit_intercept is True the shape is (order + 1,) with the intercept last.

    Raises:
        ValueError: if no sequence is long enough to yield a single window.
    """
    design_rows, targets = [], []
    for value_sequence in value_sequences:
        value_sequence = np.asarray(value_sequence, dtype=np.float64)
        for t in range(order, len(value_sequence)):
            # Reversed so column 0 is always the most recent lag.
            design_rows.append(value_sequence[t - order:t][::-1])
            targets.append(value_sequence[t])
    if not design_rows:
        raise ValueError(
            f"no sequence long enough for order {order}: need at least "
            f"{order + 1} values in some sequence")
    design_matrix = np.asarray(design_rows)
    targets = np.asarray(targets)
    if fit_intercept:
        design_matrix = np.hstack(
            [design_matrix, np.ones((len(design_matrix), 1))])
    normal_matrix = (design_matrix.T @ design_matrix
                     + ridge_penalty * np.eye(design_matrix.shape[1]))
    right_hand_side = design_matrix.T @ targets
    # Least squares rather than a direct solve: early in training the value
    # signal is nearly constant, which makes the lagged columns collinear and
    # the normal matrix singular. lstsq returns the minimum-norm solution there
    # instead of raising, so a checkpoint probe cannot abort a training run.
    solution, *_ = np.linalg.lstsq(normal_matrix, right_hand_side, rcond=None)
    return solution


def predict_one_step_ahead(coefficients, value_sequence, fit_intercept=False):
    """One-step-ahead predictions made from the TRUE history at every step.

    Each prediction sees the real previous values, so errors never accumulate.
    This measures how well the recurrence fits locally, and is the optimistic
    counterpart to forecast_free_running.

    Returns an array aligned with value_sequence[order:], i.e. it is
    `len(value_sequence) - order` long.
    """
    if fit_intercept:
        lag_weights, intercept = coefficients[:-1], coefficients[-1]
    else:
        lag_weights, intercept = coefficients, 0.0
    order = len(lag_weights)
    value_sequence = np.asarray(value_sequence, dtype=np.float64)
    return np.asarray([
        lag_weights @ value_sequence[t - order:t][::-1] + intercept
        for t in range(order, len(value_sequence))
    ])


def forecast_free_running(coefficients, seed_values, horizon,
                          fit_intercept=False):
    """Recursive multi-step forecast: predictions are fed back as history.

    After the first `order` seed values the recurrence runs on its own output,
    so errors compound. This is the honest test of the low-rank claim: a model
    can look excellent one step ahead and still diverge here.

    Args:
        coefficients: as returned by fit_autoregressive_coefficients.
        seed_values: at least `order` real values; the last `order` are used.
        horizon: how many values to forecast.

    Returns:
        Array of length `horizon`.
    """
    if fit_intercept:
        lag_weights, intercept = coefficients[:-1], coefficients[-1]
    else:
        lag_weights, intercept = coefficients, 0.0
    order = len(lag_weights)
    history = list(np.asarray(seed_values, dtype=np.float64)[-order:])
    if len(history) < order:
        raise ValueError(
            f"need at least {order} seed values, got {len(history)}")
    forecast = []
    for _ in range(horizon):
        next_value = lag_weights @ np.asarray(history[-order:][::-1]) + intercept
        forecast.append(next_value)
        history.append(next_value)
    return np.asarray(forecast)


def forecast_rolling_horizon(coefficients, value_sequence, horizon,
                             fit_intercept=False):
    """Repeated `horizon`-step forecasts, re-anchored on the true values.

    The sequence is walked in blocks of `horizon`. At the start of each block
    the recurrence is seeded with the `order` values that TRULY precede it, then
    runs free for `horizon` steps. The next block re-seeds from the real values
    again, so errors compound within a block but never across one.

    This is the regime that matches how such a model would actually be used:
    you forecast a little way ahead, then you observe what really happened and
    forecast again from there. It also interpolates cleanly between the two
    extremes already available:

        horizon = 1                     identical to predict_one_step_ahead
        horizon >= len(sequence)-order  identical to forecast_free_running

    so sweeping `horizon` measures exactly how far ahead the recurrence can see
    before it stops being useful.

    Returns an array aligned with value_sequence[order:], the same alignment
    predict_one_step_ahead uses, so the two are directly comparable.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    order = len(coefficients) - 1 if fit_intercept else len(coefficients)
    value_sequence = np.asarray(value_sequence, dtype=np.float64)
    n_predictions = len(value_sequence) - order
    if n_predictions <= 0:
        return np.zeros(0)
    predictions = np.empty(n_predictions)
    for block_start in range(0, n_predictions, horizon):
        # The `order` true values immediately preceding this block.
        seed_end = order + block_start
        seed_values = value_sequence[seed_end - order:seed_end]
        block_length = min(horizon, n_predictions - block_start)
        predictions[block_start:block_start + block_length] = \
            forecast_free_running(coefficients, seed_values, block_length,
                                  fit_intercept=fit_intercept)
    return predictions


def root_mean_squared_error(predicted, actual):
    """Plain RMSE, in the same units as the value signal."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def normalised_root_mean_squared_error(predicted, actual):
    """RMSE divided by the root-mean-square of the true signal.

    Scale-free, so it is comparable across environments whose value magnitudes
    differ by orders of magnitude (Acrobot values are large and negative,
    CartPole values are small and positive). A value of 1.0 means the error is
    as large as the signal itself.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    signal_root_mean_square = np.sqrt(np.mean(np.asarray(actual) ** 2))
    return float(root_mean_squared_error(predicted, actual)
                 / max(signal_root_mean_square, 1e-12))
