"""RunLogger.log_hankel_sweep: the sv_01..sv_NN singular-value columns
(SWEEP_N_SV) appended to hankel_sweep.csv — stable width, nan-padded, and
backward compatible with callers that pass no spectrum — plus the one-SVD
rank.spectrum_and_metrics helper the Hankel sweep feeds them from."""
import csv
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from analysis.low_rank import rank                                  # noqa: E402
from analysis.run_logger import RunLogger                           # noqa: E402

LEGACY = ["episode", "matrix", "rollout", "seed", "sub_len",
          "eff_rank", "stable_rank", "spikiness", "n_rows", "n_cols",
          "nnz_rows", "nnz_cols", "row_coherence", "col_coherence",
          "row_lev_min", "row_lev_max", "col_lev_min", "col_lev_max"]


def _metrics():
    irs = np.array([0.2, 0.8])
    ics = np.array([0.5, 0.5])
    return (3, 1.5, 1.2, (4, 4), irs, ics, 1.1, 1.3, 4, 4)


def test_log_hankel_sweep_sv_columns(tmp_path):
    logger = RunLogger(tmp_path, run_id="sweep")
    n = RunLogger.SWEEP_N_SV
    assert n == 12
    logger.log_hankel_sweep(0, "q", 0, 52, 8, *_metrics(),
                            s_vals=np.array([3.0, 2.0, 1.0]))
    logger.log_hankel_sweep(0, "q", 0, 52, 16, *_metrics())      # legacy caller
    logger.log_hankel_sweep(1, "q", 1, 53, 8, *_metrics(),
                            s_vals=np.arange(20, 0, -1.0))         # longer than n
    with open(logger.dir / "hankel_sweep.csv") as f:
        rows = list(csv.reader(f))
    sv_cols = [f"sv_{j + 1:02d}" for j in range(n)]
    assert rows[0] == LEGACY + sv_cols
    assert all(len(r) == len(rows[0]) for r in rows[1:])          # fixed width
    body = [dict(zip(rows[0], r)) for r in rows[1:]]
    # legacy columns unchanged
    assert body[0]["episode"] == "0" and body[0]["matrix"] == "q"
    assert body[0]["sub_len"] == "8" and body[0]["eff_rank"] == "3"
    assert body[0]["stable_rank"] == "1.5000" and body[0]["row_lev_min"] == "0.2"
    # short spectrum: values then nan padding
    assert [float(body[0][c]) for c in sv_cols[:3]] == [3.0, 2.0, 1.0]
    assert all(math.isnan(float(body[0][c])) for c in sv_cols[3:])
    # no spectrum at all: every sv column nan
    assert all(math.isnan(float(body[1][c])) for c in sv_cols)
    # long spectrum: truncated to the first n
    assert [float(body[2][c]) for c in sv_cols] == list(range(20, 8, -1))
    assert "sv_13" not in rows[0]


def test_spectrum_and_metrics_is_one_svd():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 9))  # rank 3
    s_vals, metrics = rank.spectrum_and_metrics(m)
    assert s_vals.shape == (6,)
    assert np.all(np.diff(s_vals) <= 0)                     # descending
    assert np.allclose(s_vals, np.linalg.svd(m, compute_uv=False))
    ref = rank.compute_rank_metrics(m)
    assert len(metrics) == len(ref) == 10
    for a, b in zip(metrics, ref):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert metrics[0] == 3                                  # energy rank


class _StubAgent:
    """The minimal QAgent surface collect_hankel_sequences reads: a policy_net
    mapping (B, 4) -> (B, 2), a greedy pi(state) and a device."""

    def __init__(self):
        import torch
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.policy_net = torch.nn.Linear(4, 2)

    def pi(self, state):
        import torch
        with torch.no_grad():
            return int(self.policy_net(
                torch.as_tensor(state, dtype=torch.float32)).argmax())


def test_hankel_sweep_analysis_populates_sv_columns(tmp_path):
    """End to end through hankel_policy.hankel_sweep_analysis: every sweep
    row carries a sorted, finite leading spectrum (nan beyond min(m, n)),
    consistent with the stable rank logged beside it, and the milestone
    spectrum figure still renders from the shared SVD."""
    import gymnasium as gym
    from analysis.low_rank import hankel_policy

    logger = RunLogger(tmp_path, run_id="e2e")
    env = gym.make("CartPole-v1")
    cfg = {"n_rollouts": 2, "base_seed": 3, "functions": ["Hankel Q"],
           "save_trajectories": False,
           "sub_trajectory": {"enabled": True, "min_len": 6, "stride": 4,
                              "n_figures": 1}}
    hankel_policy.hankel_sweep_analysis(_StubAgent(), env, cfg,
                                        run_logger=logger, episode=7)
    env.close()
    with open(logger.dir / "hankel_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows and {r["rollout"] for r in rows} == {"0", "1"}
    n = RunLogger.SWEEP_N_SV
    for r in rows:
        assert r["episode"] == "7" and r["matrix"] == "Hankel Q"
        m, k = int(r["n_rows"]), int(r["n_cols"])
        n_sv = min(m, k, n)
        vals = [float(r[f"sv_{j + 1:02d}"]) for j in range(n)]
        assert all(np.isfinite(vals[:n_sv]))
        assert all(math.isnan(v) for v in vals[n_sv:])
        assert vals[:n_sv] == sorted(vals[:n_sv], reverse=True)
        if min(m, k) <= n:            # the full spectrum fits: cross-check
            assert sum(v * v for v in vals[:n_sv]) / vals[0] ** 2 == \
                pytest.approx(float(r["stable_rank"]), abs=2e-4)
    assert any(p.suffix == ".png" for p in logger.figures_dir.iterdir())
