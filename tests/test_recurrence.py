"""Tests for analysis.low_rank.recurrence.

These pin the primitives against sequences whose true recurrence is known in
closed form, so a regression shows up as a coefficient that no longer matches
the generating process rather than merely as a worse fit.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analysis.low_rank.recurrence import (fit_autoregressive_coefficients,
                                          forecast_free_running,
                                          forecast_rolling_horizon,
                                          one_minus_r_squared,
                                          predict_one_step_ahead,
                                          root_mean_squared_error)


def test_recovers_geometric_decay():
    """A pure geometric decay is an exact order-1 recurrence."""
    value_sequence = 5.0 * 0.97 ** np.arange(200)
    coefficients = fit_autoregressive_coefficients([value_sequence], order=1)
    assert abs(coefficients[0] - 0.97) < 1e-8
    forecast = forecast_free_running(coefficients, value_sequence[:1],
                                     horizon=199)
    assert np.allclose(forecast, value_sequence[1:], atol=1e-8)


def test_recovers_order_two_oscillation():
    """A damped oscillation is an exact order-2 recurrence with known roots."""
    damping, angle = 0.98, 0.3
    true_coefficients = np.array([2 * damping * np.cos(angle), -damping ** 2])
    value_sequence = np.zeros(300)
    value_sequence[0], value_sequence[1] = 1.0, 0.9
    for t in range(2, 300):
        value_sequence[t] = true_coefficients @ np.array(
            [value_sequence[t - 1], value_sequence[t - 2]])
    coefficients = fit_autoregressive_coefficients([value_sequence], order=2)
    assert np.allclose(coefficients, true_coefficients, atol=1e-6)
    forecast = forecast_free_running(coefficients, value_sequence[:2],
                                     horizon=100)
    assert one_minus_r_squared(
        forecast, value_sequence[2:102]) < 1e-6


def test_single_global_fit_across_many_sequences():
    """One global fit over several trajectories must recover the shared
    recurrence, which is what the probe relies on when it pools trajectories."""
    true_coefficients = np.array([1.6, -0.64])  # (1 - 0.8 z^-1)^2, double root
    generator = np.random.default_rng(0)
    value_sequences = []
    for _ in range(5):
        value_sequence = np.zeros(80)
        value_sequence[0], value_sequence[1] = generator.normal(size=2)
        for t in range(2, 80):
            value_sequence[t] = true_coefficients @ np.array(
                [value_sequence[t - 1], value_sequence[t - 2]])
        value_sequences.append(value_sequence)
    coefficients = fit_autoregressive_coefficients(value_sequences, order=2)
    assert np.allclose(coefficients, true_coefficients, atol=1e-5)

    held_out = np.zeros(60)
    held_out[0], held_out[1] = 2.0, 1.5
    for t in range(2, 60):
        held_out[t] = true_coefficients @ np.array(
            [held_out[t - 1], held_out[t - 2]])
    assert one_minus_r_squared(
        forecast_free_running(coefficients, held_out[:2], 58), held_out[2:]) < 1e-5


def test_one_step_alignment_and_noise_stability():
    generator = np.random.default_rng(1)
    value_sequence = (np.sin(0.2 * np.arange(150))
                      + 0.05 * generator.normal(size=150))
    coefficients = fit_autoregressive_coefficients(
        [value_sequence], order=4, ridge_penalty=1e-6)
    predictions = predict_one_step_ahead(coefficients, value_sequence)
    assert predictions.shape == (146,), "must align with value_sequence[order:]"
    assert np.isfinite(predictions).all()
    assert one_minus_r_squared(predictions,
                                              value_sequence[4:]) < 0.5
    assert np.isfinite(forecast_free_running(coefficients, value_sequence[:4],
                                             horizon=50)).all()


def test_rolling_horizon_of_one_equals_one_step_ahead():
    """horizon=1 must reproduce predict_one_step_ahead exactly: each block is a
    single step seeded from the true history."""
    generator = np.random.default_rng(5)
    value_sequence = np.cumsum(generator.normal(size=120)) + 10.0
    coefficients = fit_autoregressive_coefficients([value_sequence], order=3)
    assert np.allclose(
        forecast_rolling_horizon(coefficients, value_sequence, horizon=1),
        predict_one_step_ahead(coefficients, value_sequence))


def test_rolling_horizon_covering_whole_sequence_equals_free_running():
    """A horizon at least as long as the sequence leaves one block, which is
    exactly the free-running forecast from the start."""
    generator = np.random.default_rng(6)
    value_sequence = np.cumsum(generator.normal(size=90)) + 5.0
    order = 4
    coefficients = fit_autoregressive_coefficients([value_sequence], order=order)
    rolling = forecast_rolling_horizon(coefficients, value_sequence,
                                       horizon=len(value_sequence))
    free_running = forecast_free_running(coefficients, value_sequence[:order],
                                         horizon=len(value_sequence) - order)
    assert np.allclose(rolling, free_running)


def test_rolling_horizon_re_anchors_on_true_values():
    """Within a block the forecast drifts; at every block boundary it must snap
    back to a prediction made from the real history."""
    order, horizon = 2, 8
    # A sequence the recurrence cannot follow perfectly, so drift is visible.
    generator = np.random.default_rng(7)
    value_sequence = (np.sin(0.35 * np.arange(120))
                      + 0.3 * generator.normal(size=120) + 4.0)
    coefficients = fit_autoregressive_coefficients([value_sequence], order=order,
                                                   ridge_penalty=1e-6)
    rolling = forecast_rolling_horizon(coefficients, value_sequence,
                                       horizon=horizon)
    one_step = predict_one_step_ahead(coefficients, value_sequence)
    # The first prediction of every block is seeded entirely from true values,
    # so it must equal the one-step-ahead prediction at that index.
    for block_start in range(0, len(rolling), horizon):
        assert abs(rolling[block_start] - one_step[block_start]) < 1e-9, \
            f"block at {block_start} did not re-anchor on the true history"
    # And a longer horizon must be at least as hard as a shorter one.
    error_short = one_minus_r_squared(
        forecast_rolling_horizon(coefficients, value_sequence, 2),
        value_sequence[order:])
    error_long = one_minus_r_squared(
        forecast_rolling_horizon(coefficients, value_sequence, 32),
        value_sequence[order:])
    assert error_long > error_short


def test_rolling_horizon_alignment_and_short_sequences():
    coefficients = fit_autoregressive_coefficients([np.arange(30.0)], order=3)
    assert forecast_rolling_horizon(coefficients, np.arange(30.0),
                                    horizon=7).shape == (27,)
    # Sequence not longer than the order yields no predictions rather than raising.
    assert forecast_rolling_horizon(coefficients, np.arange(3.0),
                                    horizon=7).shape == (0,)
    try:
        forecast_rolling_horizon(coefficients, np.arange(30.0), horizon=0)
        raise AssertionError("expected ValueError for horizon 0")
    except ValueError as error:
        assert "at least 1" in str(error)


def test_intercept_handles_constant_offset():
    """Value sequences rarely decay to zero, so the intercept term matters."""
    value_sequence = 3.0 + 2.0 * 0.9 ** np.arange(100)
    coefficients = fit_autoregressive_coefficients(
        [value_sequence], order=1, fit_intercept=True)
    forecast = forecast_free_running(coefficients, value_sequence[:1],
                                     horizon=99, fit_intercept=True)
    assert one_minus_r_squared(forecast,
                                              value_sequence[1:]) < 1e-6


def test_fit_rejects_sequences_that_are_too_short():
    try:
        fit_autoregressive_coefficients([np.arange(3.0)], order=8)
        raise AssertionError("expected ValueError for too-short sequence")
    except ValueError as error:
        assert "order 8" in str(error)


def test_one_minus_r_squared_is_scale_and_offset_free():
    """It must be invariant to rescaling AND to a constant offset.

    Offset invariance is the whole point of moving off the RMS normaliser: a
    value signal sits at a large non-zero level, and a metric that changes when
    you shift that level is measuring the level rather than the dynamics."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 2.1, 2.9, 4.2])
    plain = root_mean_squared_error(predicted, actual)
    assert abs(root_mean_squared_error(100 * predicted, 100 * actual)
               - 100 * plain) < 1e-9, "plain RMSE scales with the signal"
    base = one_minus_r_squared(predicted, actual)
    assert abs(one_minus_r_squared(100 * predicted, 100 * actual) - base) < 1e-12
    assert abs(one_minus_r_squared(predicted + 500, actual + 500) - base) < 1e-12


def test_one_minus_r_squared_reference_points():
    """The scale must mean what it says: 0 perfect, 1 = predicting the mean."""
    generator = np.random.default_rng(21)
    actual = generator.normal(size=400) * 3.0 + 50.0
    assert one_minus_r_squared(actual, actual) == 0.0
    # predicting the signal's own mean is exactly 1.0 by construction
    mean_prediction = np.full_like(actual, actual.mean())
    assert abs(one_minus_r_squared(mean_prediction, actual) - 1.0) < 1e-12
    # a worse-than-mean prediction goes above 1
    assert one_minus_r_squared(np.full_like(actual, actual.mean() + 10), actual) > 1.0
    # constant signal has no variance to explain
    assert np.isnan(one_minus_r_squared(np.ones(5), np.ones(5)))


def test_one_minus_r_squared_is_not_flattered_by_a_large_mean():
    """The failure that motivated the change: on a signal with a big offset and
    small variation, the RMS-normalised error looks tiny while 1 - R^2 does not."""
    actual = np.sin(np.linspace(0, 20, 300)) * 0.03 - 1.56   # Acrobot-like
    mean_prediction = np.full_like(actual, actual.mean())
    rms_normalised = (root_mean_squared_error(mean_prediction, actual)
                      / np.sqrt(np.mean(actual ** 2)))
    assert rms_normalised < 0.02, "RMS normaliser makes a useless model look great"
    assert abs(one_minus_r_squared(mean_prediction, actual) - 1.0) < 1e-9, \
        "1 - R^2 correctly calls it no better than the mean"


if __name__ == "__main__":
    test_functions = [value for name, value in sorted(globals().items())
                      if name.startswith("test_")]
    for test_function in test_functions:
        test_function()
        print(f"PASS {test_function.__name__}")
    print(f"{len(test_functions)} tests passed")
