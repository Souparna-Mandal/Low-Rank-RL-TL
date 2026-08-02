"""Tests for analysis.low_rank.recurrence (AR fitting / free-run prediction)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analysis.low_rank.recurrence import fit_ar, free_run, nrmse, predict_one_step


def test_recovers_geometric():
    s = 5.0 * 0.97 ** np.arange(200)
    c = fit_ar([s], order=1)
    assert abs(c[0] - 0.97) < 1e-8
    pred = free_run(c, s[:1], horizon=199)
    assert np.allclose(pred, s[1:], atol=1e-8)


def test_recovers_order2_oscillation():
    rho, th = 0.98, 0.3
    c_true = np.array([2 * rho * np.cos(th), -rho ** 2])
    s = np.zeros(300)
    s[0], s[1] = 1.0, 0.9
    for t in range(2, 300):
        s[t] = c_true @ np.array([s[t - 1], s[t - 2]])
    c = fit_ar([s], order=2)
    assert np.allclose(c, c_true, atol=1e-6)
    pred = free_run(c, s[:2], horizon=100)
    assert nrmse(pred, s[2:102]) < 1e-6


def test_global_fit_across_sequences():
    c_true = np.array([1.6, -0.64])  # (1 - 0.8 z^-1)^2: double real root
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(5):
        s = np.zeros(80)
        s[0], s[1] = rng.normal(size=2)
        for t in range(2, 80):
            s[t] = c_true @ np.array([s[t - 1], s[t - 2]])
        seqs.append(s)
    c = fit_ar(seqs, order=2)
    assert np.allclose(c, c_true, atol=1e-5)
    held = np.zeros(60)
    held[0], held[1] = 2.0, 1.5
    for t in range(2, 60):
        held[t] = c_true @ np.array([held[t - 1], held[t - 2]])
    assert nrmse(free_run(c, held[:2], 58), held[2:]) < 1e-5


def test_one_step_alignment_and_noise_stability():
    rng = np.random.default_rng(1)
    s = np.sin(0.2 * np.arange(150)) + 0.05 * rng.normal(size=150)
    c = fit_ar([s], order=4, ridge=1e-6)
    p = predict_one_step(c, s)
    assert p.shape == (146,)
    assert np.isfinite(p).all() and nrmse(p, s[4:]) < 0.5
    fr = free_run(c, s[:4], horizon=50)
    assert np.isfinite(fr).all()


def test_intercept_handles_offset():
    s = 3.0 + 2.0 * 0.9 ** np.arange(100)
    c = fit_ar([s], order=1, intercept=True)
    pred = free_run(c, s[:1], horizon=99, intercept=True)
    assert nrmse(pred, s[1:]) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
