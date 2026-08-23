"""Sanity tests for the Hankel construction and the rank metrics built on it.
Run as `pytest tests/test_hankel_rank.py -v -s` from repo root (-s to see the
measured ranks; `-rP` shows the same output without disabling capture)."""
import inspect
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analysis.low_rank.hankel_policy import _hankel_from_sequence
from analysis.low_rank.rank import (_svd_and_metrics, compute_rank_metrics,
                                    energy_rank, row_rank_property_check)

# Distinct non-zero modes: tau_t = sum_k lam_k^t obeys an order-R linear
# recurrence, so its Hankel matrix has rank exactly R (given min(m,n) >= R).
LAMBDAS = (1.0, 0.9, -0.85, 0.75, 0.6)
RANKS = (1, 2, 3, 4, 5)
H = 41

# energy_frac at which energy_rank stops truncating and reports the exact rank.
# Only energy_rank takes the fraction; the metrics helpers report its default.
FULL_ENERGY = 1 - 1e-9


def exponential_sequence(R, length=H):
    lam = np.asarray(LAMBDAS[:R], dtype=float)[:, None]
    return (lam ** np.arange(length)).sum(axis=0)


def random_sequence(length=H, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(length, generator=g, dtype=torch.float64).numpy()


def expected_shape(length):
    mid = length // 2
    return (mid + 1, length - mid)


def report(title, **measured):
    """Measured values for the case under test, one line, shown with -s / -rP."""
    body = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in measured.items())
    print(f"\n[{title}] {body}")


def check(claim, ok):
    """Print the assertion (with its values baked into `claim`) and then make it."""
    print(f"    {'PASS' if ok else 'FAIL'}  {claim}")
    assert ok, claim


def singular_values(hk):
    return np.linalg.svd(hk, compute_uv=False)


def spectrum(hk, k=6):
    return np.array2string(singular_values(hk)[:k], precision=3)


@pytest.mark.parametrize("length", [40, 41])
def test_hankel_from_sequence_is_a_hankel_matrix(length):
    tau = random_sequence(length, seed=3)
    hk = _hankel_from_sequence(tau)
    m, n = hk.shape
    i, j = np.indices((m, n))
    max_dev = float(np.abs(hk - tau[i + j]).max())

    report(f"structure H={length}", mid=length // 2, shape=hk.shape,
           expected=expected_shape(length), max_antidiag_dev=max_dev)
    check(f"shape {hk.shape} == expected {expected_shape(length)}",
          hk.shape == expected_shape(length))
    check(f"hk[i,j] == tau[i+j] everywhere (max dev {max_dev:.3g})",
          np.allclose(hk, tau[i + j]))


@pytest.mark.parametrize("R", RANKS)
def test_exponential_sequence_has_predicted_rank(R):
    """A sum of R exponentials -> Hankel of rank exactly R."""
    hk = _hankel_from_sequence(exponential_sequence(R))
    true_rank = int(np.linalg.matrix_rank(hk))
    exact_energy_rank = energy_rank(singular_values(hk), FULL_ENERGY)
    eff_rank, stable_rank = compute_rank_metrics(hk)[:2]

    report(f"exponential R={R}", lambdas=LAMBDAS[:R], shape=hk.shape,
           matrix_rank=true_rank, energy_rank_at_full=exact_energy_rank,
           reported_eff_rank=eff_rank, stable_rank=float(stable_rank),
           top_svals=spectrum(hk))
    check(f"matrix is big enough for rank {R}: min{hk.shape} > {R}",
          min(hk.shape) > R)
    check(f"numerical rank {true_rank} == predicted {R}", true_rank == R)
    check(f"energy_rank @ energy_frac={FULL_ENERGY} gives "
          f"{exact_energy_rank} == {R}", exact_energy_rank == R)
    # the reported (0.999) rank never over-reports, and stable rank is bounded by the true rank
    check(f"reported eff rank {eff_rank} in [1, {R}]", 1 <= eff_rank <= R)
    check(f"stable rank {stable_rank:.4f} in [1, {R}]",
          1 <= stable_rank <= R + 1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_sequence_hankel_is_full_rank(seed):
    hk = _hankel_from_sequence(random_sequence(seed=seed))
    full = min(hk.shape)
    true_rank = int(np.linalg.matrix_rank(hk))
    exact_energy_rank = energy_rank(singular_values(hk), FULL_ENERGY)
    eff_rank, stable_rank = compute_rank_metrics(hk)[:2]

    report(f"random seed={seed}", shape=hk.shape, full_rank=full,
           matrix_rank=true_rank, energy_rank_at_full=exact_energy_rank,
           reported_eff_rank=eff_rank, stable_rank=float(stable_rank),
           top_svals=spectrum(hk))
    check(f"numerical rank {true_rank} == full {full}", true_rank == full)
    check(f"energy_rank @ energy_frac={FULL_ENERGY} gives "
          f"{exact_energy_rank} == {full}", exact_energy_rank == full)
    # energy genuinely spread over the spectrum, unlike the exponential case
    check(f"reported eff rank {eff_rank} > {full // 4}", eff_rank > full // 4)
    check(f"stable rank {stable_rank:.4f} > 2.0", stable_rank > 2.0)


def test_random_sequence_ranks_above_low_rank_sequence():
    rand_rank, rand_stable = compute_rank_metrics(
        _hankel_from_sequence(random_sequence(seed=7)))[:2]
    low_rank, low_stable = compute_rank_metrics(
        _hankel_from_sequence(exponential_sequence(3)))[:2]

    report("random vs low-rank", random_eff_rank=rand_rank,
           random_stable=float(rand_stable), exponential_eff_rank=low_rank,
           exponential_stable=float(low_stable))
    check(f"reported eff rank: random {rand_rank} > exponential {low_rank}",
          rand_rank > low_rank)
    check(f"stable rank: random {rand_stable:.4f} > exponential {low_stable:.4f}",
          rand_stable > low_stable)


@pytest.mark.parametrize("tau", [exponential_sequence(3), random_sequence(seed=5)],
                         ids=["exponential", "random"])
def test_rank_metrics_invariants(tau):
    hk = _hankel_from_sequence(tau)
    (rank, stable_rank, spikiness, shape, irs, ics,
     row_coh, col_coh, nnz_rows, nnz_cols) = compute_rank_metrics(hk)
    m, n = shape
    true_rank = int(np.linalg.matrix_rank(hk))
    nnz = (int(nnz_rows), int(nnz_cols))

    report("metrics", shape=shape, matrix_rank=true_rank, reported_eff_rank=rank,
           stable_rank=float(stable_rank), spikiness=float(spikiness),
           row_coherence=float(row_coh), col_coherence=float(col_coh),
           irs_sum=float(irs.sum()), ics_sum=float(ics.sum()), nnz=nnz)
    check(f"reported shape {shape} == {hk.shape}", shape == hk.shape)
    check(f"eff rank {rank} in [1, {min(m, n)}]", 1 <= rank <= min(m, n))
    check(f"stable rank {stable_rank:.4f} in [1, matrix_rank {true_rank}]",
          1 <= stable_rank <= true_rank + 1e-9)
    check(f"spikiness {spikiness:.4f} >= 1", spikiness >= 1.0)
    check(f"leverage shares sized {irs.shape}, {ics.shape} == ({m},), ({n},)",
          irs.shape == (m,) and ics.shape == (n,))
    check(f"leverage shares sum to 1: {irs.sum():.6f}, {ics.sum():.6f}",
          np.isclose(irs.sum(), 1.0) and np.isclose(ics.sum(), 1.0))
    check(f"row coherence {row_coh:.4f} in [1, m/rank {m / rank:.4f}]",
          1 - 1e-9 <= row_coh <= m / rank + 1e-9)
    check(f"col coherence {col_coh:.4f} in [1, n/rank {n / rank:.4f}]",
          1 - 1e-9 <= col_coh <= n / rank + 1e-9)
    check(f"dense sequence: nnz rows/cols {nnz} == {(m, n)}", nnz == (m, n))


def test_energy_frac_lives_only_on_energy_rank():
    """The reported rank is energy_rank's default (0.999) and nothing downstream
    carries its own copy of the fraction to drift out of sync."""
    hk = _hankel_from_sequence(random_sequence(seed=11))
    default_frac = inspect.signature(energy_rank).parameters["energy_frac"].default
    expected = energy_rank(singular_values(hk), default_frac)
    propagated = [f.__name__ for f in (_svd_and_metrics, compute_rank_metrics,
                                       row_rank_property_check)
                  if "energy_frac" in inspect.signature(f).parameters]

    report("default energy_frac", default=default_frac, expected_rank=expected,
           reported_rank=compute_rank_metrics(hk)[0], propagated_to=propagated)
    check(f"energy_rank default {default_frac} == 0.999", default_frac == 0.999)
    check(f"compute_rank_metrics reports energy_rank's default: "
          f"{compute_rank_metrics(hk)[0]} == {expected}",
          compute_rank_metrics(hk)[0] == expected)
    check(f"no downstream copy of energy_frac (found in {propagated})",
          propagated == [])


def test_energy_rank_truncates_by_energy():
    s = np.array([10.0, 1.0, 0.1])                      # energies 100, 1, 0.01
    ranks = {f: energy_rank(s, f) for f in (0.90, 0.999, FULL_ENERGY)}

    report("energy_rank", svals=s, energies=s**2, ranks=ranks,
           zero_spectrum=energy_rank(np.zeros(4)))
    check(f"90% energy -> {ranks[0.90]} == 1", ranks[0.90] == 1)
    check(f"99.9% energy -> {ranks[0.999]} == 2", ranks[0.999] == 2)
    check(f"{FULL_ENERGY} energy -> {ranks[FULL_ENERGY]} == 3",
          ranks[FULL_ENERGY] == 3)
    check(f"all-zero spectrum -> {energy_rank(np.zeros(4))} == 0",
          energy_rank(np.zeros(4)) == 0)
