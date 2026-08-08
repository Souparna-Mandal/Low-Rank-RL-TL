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
    example_rollout_arrays, metric_rows)


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
