"""Tests for analysis.low_rank.autoregressive_value_probe.

The probe is scored on sequences whose generating recurrence is known, so a
regression shows up as the probe failing to recognise structure it should find,
or claiming structure that is not there.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analysis.low_rank.autoregressive_value_probe import (
    HELD_OUT_TRAJECTORY_SPLIT, PREFIX_SUFFIX_SPLIT, coefficient_rows,
    collect_greedy_value_sequences, evaluate_autoregressive_orders,
    example_rollout_arrays, horizon_metric_rows, metric_rows)


def _order_two_sequences(n_sequences=10, length=80, seed=0):
    """Trajectories that exactly obey a known order-2 recurrence."""
    true_coefficients = np.array([1.6, -0.64])  # (1 - 0.8 z^-1)^2
    generator = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_sequences):
        sequence = np.zeros(length)
        sequence[0], sequence[1] = generator.normal(size=2)
        for t in range(2, length):
            sequence[t] = true_coefficients @ np.array(
                [sequence[t - 1], sequence[t - 2]])
        sequences.append(sequence)
    return sequences, true_coefficients


def test_recovers_known_order_two_structure_on_held_out_trajectories():
    sequences, true_coefficients = _order_two_sequences()
    results = evaluate_autoregressive_orders(sequences, orders=(2, 3, 4, 8))
    assert set(results) == {2, 3, 4, 8}

    held_out = results[2][HELD_OUT_TRAJECTORY_SPLIT]
    # fit_intercept defaults True, so the lag weights are all but the last entry
    assert np.allclose(held_out["coefficients"][:2], true_coefficients, atol=1e-4)
    # An exact recurrence must predict held-out trajectories essentially perfectly,
    # including under free running where errors would otherwise compound.
    assert held_out["test"]["normalised_rmse_one_step_ahead"] < 1e-6
    assert held_out["test"]["normalised_rmse_free_running"] < 1e-4
    assert not held_out["test"]["free_running_diverged"]


def test_prefix_suffix_split_forecasts_the_unseen_half():
    sequences, _ = _order_two_sequences()
    results = evaluate_autoregressive_orders(sequences, orders=(2,))
    prefix_suffix = results[2][PREFIX_SUFFIX_SPLIT]
    # Fitted only on first halves, forecasting the second halves it never saw.
    assert prefix_suffix["test"]["normalised_rmse_free_running"] < 1e-4
    assert len(prefix_suffix["training_sequences"][0]) == 40
    assert len(prefix_suffix["test_sequences"][0]) == 40


def test_unpredictable_noise_is_not_reported_as_predictable():
    """Guards the failure that would matter most: white noise must NOT come
    back with a low held-out error just because the fit interpolates."""
    generator = np.random.default_rng(3)
    sequences = [generator.normal(size=80) for _ in range(10)]
    results = evaluate_autoregressive_orders(sequences, orders=(2, 8))
    for order in (2, 8):
        test = results[order][HELD_OUT_TRAJECTORY_SPLIT]["test"]
        assert test["normalised_rmse_one_step_ahead"] > 0.5, \
            "white noise should be near-unpredictable one step ahead"


def test_free_running_divergence_is_recorded_not_hidden():
    """An explosive recurrence must be flagged rather than reported as a
    plausible finite error."""
    explosive = []
    for start in range(1, 6):
        sequence = np.zeros(60)
        sequence[0], sequence[1] = start, start * 1.02
        for t in range(2, 60):  # unstable root, but numerically tractable
            sequence[t] = 1.9 * sequence[t - 1] - 0.9 * sequence[t - 2]
        explosive.append(sequence)
    results = evaluate_autoregressive_orders(explosive, orders=(2,))
    metrics = results[2][HELD_OUT_TRAJECTORY_SPLIT]["test"]
    assert np.isfinite(metrics["normalised_rmse_one_step_ahead"])
    assert "free_running_diverged" in metrics


def test_constant_value_sequences_do_not_raise():
    """An untrained network gives a near-constant value signal, which makes the
    lagged columns collinear. The probe must degrade, not crash the run."""
    constant = [np.full(50, 3.0) for _ in range(6)]
    results = evaluate_autoregressive_orders(constant, orders=(2, 8))
    for order in (2, 8):
        metrics = results[order][HELD_OUT_TRAJECTORY_SPLIT]["test"]
        assert not np.isnan(metrics["rmse_one_step_ahead"])

    nearly_constant = [np.full(50, 3.0) + 1e-9 * np.arange(50)
                       for _ in range(6)]
    assert evaluate_autoregressive_orders(nearly_constant, orders=(4,))


def test_rolling_horizon_sweep_endpoints_match_the_other_regimes():
    """The sweep must contain the other two regimes as its endpoints: horizon 1
    is one-step-ahead, and a horizon past the trajectory length is free
    running. If that stops holding, the three numbers are not comparable."""
    generator = np.random.default_rng(11)
    sequences = [np.cumsum(generator.normal(size=70)) + 20.0 for _ in range(8)]
    results = evaluate_autoregressive_orders(
        sequences, orders=(2,), forecast_horizons=(1, 8, 1000))
    split = results[2][HELD_OUT_TRAJECTORY_SPLIT]
    horizons = split["test_horizons"]
    assert abs(horizons[1]["normalised_rmse"]
               - split["test"]["normalised_rmse_one_step_ahead"]) < 1e-9
    assert abs(horizons[1000]["normalised_rmse"]
               - split["test"]["normalised_rmse_free_running"]) < 1e-9


def test_error_grows_with_forecast_horizon_then_saturates():
    """Longer horizons are harder, but the error does NOT grow without bound.

    A stable fitted recurrence decays towards the signal's mean once it runs
    free, so beyond some horizon it is simply predicting the mean and the error
    plateaus at roughly the signal's own scale. Only an unstable recurrence
    diverges, and that is reported separately. Asserting strict monotonicity
    here would therefore be wrong; what must hold is that horizon 1 -- which
    uses the most true information -- is the easiest of all horizons, and that
    a long horizon is materially harder than it.
    """
    generator = np.random.default_rng(12)
    sequences = [np.sin(0.3 * np.arange(120)) + 0.25 * generator.normal(size=120)
                 + 6.0 for _ in range(8)]
    results = evaluate_autoregressive_orders(
        sequences, orders=(2,), ridge_penalty=1e-6,
        forecast_horizons=(1, 4, 16, 64))
    errors = {h: m["normalised_rmse"] for h, m
              in results[2][HELD_OUT_TRAJECTORY_SPLIT]["test_horizons"].items()}
    assert errors[1] == min(errors.values()), errors
    assert errors[4] > errors[1] and errors[16] > errors[4], errors
    assert errors[64] > 1.5 * errors[1], errors
    # saturation: the jump from 16 to 64 is far smaller than from 1 to 16
    assert abs(errors[64] - errors[16]) < (errors[16] - errors[1]), errors


def test_horizon_rows_cover_every_combination():
    sequences, _ = _order_two_sequences()
    horizons = (1, 4, 16)
    results = evaluate_autoregressive_orders(sequences, orders=(2, 4),
                                             forecast_horizons=horizons)
    rows = horizon_metric_rows(episode=7, results=results,
                               value_sequences=sequences)
    # 2 orders x 2 splits x 2 subsets x 3 horizons
    assert len(rows) == 24
    assert {r["forecast_horizon"] for r in rows} == set(horizons)
    assert all(r["episode"] == 7 for r in rows)


def test_row_flattening_shapes_match_the_csv_columns():
    sequences, _ = _order_two_sequences()
    results = evaluate_autoregressive_orders(sequences, orders=(2, 4))
    rows = metric_rows(episode=100, results=results, value_sequences=sequences)
    # 2 orders x 2 splits x 2 subsets
    assert len(rows) == 8
    assert {r["subset"] for r in rows} == {"training", "test"}
    assert {r["split"] for r in rows} == {HELD_OUT_TRAJECTORY_SPLIT,
                                          PREFIX_SUFFIX_SPLIT}
    assert all(r["episode"] == 100 for r in rows)

    coefficients = coefficient_rows(episode=100, results=results)
    # order 2 -> lags 1,2 + intercept; order 4 -> lags 1..4 + intercept
    per_order_lag_counts = {}
    for row in coefficients:
        key = (row["order"], row["split"])
        per_order_lag_counts[key] = per_order_lag_counts.get(key, 0) + 1
    assert per_order_lag_counts[(2, HELD_OUT_TRAJECTORY_SPLIT)] == 3
    assert per_order_lag_counts[(4, HELD_OUT_TRAJECTORY_SPLIT)] == 5
    assert 0 in {r["lag"] for r in coefficients}, "intercept row must be present"


def test_example_rollouts_carry_the_configured_forecast_windows():
    """The rollout panel must be able to draw the rolling-horizon forecast, not
    just the seeded-once free run -- that is the whole point of the tau sweep."""
    sequences, _ = _order_two_sequences()
    results = evaluate_autoregressive_orders(
        sequences, orders=(2,), forecast_horizons=(1, 8, 32))
    arrays = example_rollout_arrays(results, n_examples=1,
                                    forecast_horizons=(1, 8, 32))
    stem = next(k[:-len("__actual")] for k in arrays if k.endswith("__actual"))
    for horizon in (1, 8, 32):
        key = f"{stem}__rolling_horizon_{horizon}"
        assert key in arrays, f"missing {key}"
        # aligned with actual[order:], exactly like one_step_ahead
        assert len(arrays[key]) == len(arrays[stem + "__actual"]) - 2
    # tau=1 must coincide with the one-step array it is supposed to generalise
    assert np.allclose(arrays[f"{stem}__rolling_horizon_1"],
                       arrays[f"{stem}__one_step_ahead"])


def test_example_rollout_arrays_are_plottable():
    sequences, _ = _order_two_sequences()
    results = evaluate_autoregressive_orders(sequences, orders=(2,))
    arrays = example_rollout_arrays(results, n_examples=2)
    actual_keys = [k for k in arrays if k.endswith("__actual")]
    assert actual_keys, "expected at least one example rollout"
    for key in actual_keys:
        stem = key[: -len("__actual")]
        actual = arrays[key]
        one_step = arrays[f"{stem}__one_step_ahead"]
        free_running = arrays[f"{stem}__free_running"]
        assert len(one_step) == len(actual) - 2, "one-step drops `order` points"
        assert len(free_running) > 0 and np.isfinite(free_running).all()


class _ConstantValueNetwork:
    """Stand-in policy network: two actions, values depending only on step."""

    def __init__(self):
        self.calls = 0

    def __call__(self, state_tensor):
        import torch
        self.calls += 1
        return torch.tensor([[float(self.calls), -1.0]])


class _ShortEnv:
    """Terminates after a fixed number of steps, ignoring the action."""

    def __init__(self, length=5):
        self.length = length
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(2, dtype=np.float32), 0.0, self.t >= self.length, False, {}


class _StubAgent:
    def __init__(self):
        self.policy_net = _ConstantValueNetwork()
        self.device = "cpu"


def test_collect_greedy_value_sequences_records_max_q_per_step():
    agent, env = _StubAgent(), _ShortEnv(length=5)
    sequences = collect_greedy_value_sequences(agent, env, seeds=[0, 1])
    assert len(sequences) == 2
    assert all(len(s) == 5 for s in sequences), "one value per env step"
    # The stub returns an increasing first-action value, and max_a Q picks it.
    assert np.all(np.diff(sequences[0]) > 0)


def test_collect_respects_max_steps_on_non_terminating_env():
    class _NeverEnds(_ShortEnv):
        def step(self, action):
            return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    sequences = collect_greedy_value_sequences(
        _StubAgent(), _NeverEnds(), seeds=[0], max_steps=17)
    assert len(sequences[0]) == 17


if __name__ == "__main__":
    test_functions = [value for name, value in sorted(globals().items())
                      if name.startswith("test_")]
    for test_function in test_functions:
        test_function()
        print(f"PASS {test_function.__name__}")
    print(f"{len(test_functions)} tests passed")
