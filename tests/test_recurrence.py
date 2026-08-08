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
                                          normalised_root_mean_squared_error,
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
    assert normalised_root_mean_squared_error(
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
    assert normalised_root_mean_squared_error(
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
    assert normalised_root_mean_squared_error(predictions,
                                              value_sequence[4:]) < 0.5
    assert np.isfinite(forecast_free_running(coefficients, value_sequence[:4],
                                             horizon=50)).all()


def test_intercept_handles_constant_offset():
    """Value sequences rarely decay to zero, so the intercept term matters."""
    value_sequence = 3.0 + 2.0 * 0.9 ** np.arange(100)
    coefficients = fit_autoregressive_coefficients(
        [value_sequence], order=1, fit_intercept=True)
    forecast = forecast_free_running(coefficients, value_sequence[:1],
                                     horizon=99, fit_intercept=True)
    assert normalised_root_mean_squared_error(forecast,
                                              value_sequence[1:]) < 1e-6


def test_fit_rejects_sequences_that_are_too_short():
    try:
        fit_autoregressive_coefficients([np.arange(3.0)], order=8)
        raise AssertionError("expected ValueError for too-short sequence")
    except ValueError as error:
        assert "order 8" in str(error)


def test_normalised_error_is_scale_free():
    """The normalised error must not change when the signal is rescaled, which
    is what makes Acrobot and CartPole numbers comparable."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 2.1, 2.9, 4.2])
    plain = root_mean_squared_error(predicted, actual)
    scaled = root_mean_squared_error(100 * predicted, 100 * actual)
    assert abs(scaled - 100 * plain) < 1e-9, "plain RMSE scales with the signal"
    assert abs(normalised_root_mean_squared_error(predicted, actual)
               - normalised_root_mean_squared_error(100 * predicted,
                                                    100 * actual)) < 1e-12


if __name__ == "__main__":
    test_functions = [value for name, value in sorted(globals().items())
                      if name.startswith("test_")]
    for test_function in test_functions:
        test_function()
        print(f"PASS {test_function.__name__}")
    print(f"{len(test_functions)} tests passed")
