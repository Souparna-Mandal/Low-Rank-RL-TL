import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_spectra(svals: np.ndarray, matrix_name: str):
    # svals = np.linalg.svd(matrix, compute_uv=False)
    
    # bar plot of eigvals and log eigvals side by side
    mags = np.sort(np.abs(svals))[::-1]     # spectrum in descending magnitude
    idx = np.arange(1, len(mags) + 1)
    eps = np.finfo(mags.dtype).eps          # lower limit so log(0) doesn't blow up

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    ax_lin.bar(idx, mags, color="steelblue")
    ax_lin.set_xlabel("index")
    ax_lin.set_ylabel(r"$|\lambda|$")
    ax_lin.set_title(f"Singular Value magnitudes of {matrix_name}")

    ax_log.bar(idx, np.log10(mags + eps), color="indianred")
    ax_log.set_xlabel("index")
    ax_log.set_ylabel(r"$\log_{10}|\lambda|$")
    ax_log.set_title(f"Log Singular Value magnitudes {matrix_name}")

    fig.tight_layout()
    plt.show()
    return svals

def energy_rank(s_vals: np.ndarray, energy_frac: float = 0.90) -> int:
    """Smallest k such that the top-k singular values capture `energy_frac` of the
    matrix's Frobenius energy (sum of squared singular values, = ||A||_F^2).
    """
    s = np.sort(np.abs(np.asarray(s_vals, dtype=float)))[::-1]
    energy = s**2
    total = energy.sum()
    if total == 0:
        return 0
    cum_frac = np.cumsum(energy) / total
    return int(np.searchsorted(cum_frac, energy_frac) + 1)   # first index reaching the fraction, +1 for count

def row_rank_property_check(matrix: np.ndarray, matrix_name: str,
                            energy_frac: float = 0.90): # This should be a 2-d numpy array
    """
    We calculate for a low rank matrix the information like how much exploitable pattern is present.

    All leverage/coherence quantities are computed on the *top-r left-singular subspace* U[:, :r]
    For a (near) square matrix the full U is a complete orthonormal basis, so
    every row of it has norm 1 and the old.

    Args:
        energy_frac: the rank is the number of singular values needed to capture this
                     fraction of the Frobenius energy (see `energy_rank`). Default 0.90.
    """
    # Calculating SVD and getting thr Matrix Shape
    m,n = matrix.shape
    U, s_vals, Vt = np.linalg.svd(matrix, full_matrices=False)

    # plotting the spectrum of the Matrix
    plot_matrix_spectra(s_vals, matrix_name)

    # Effective rank: how many singular values are needed to capture `energy_frac` of the Frobenius energy.
    rank = max(energy_rank(s_vals, energy_frac), 1)
    # Stable rank: energy-based (sum sigma_i^2 / sigma_1^2), in [1, min(m,n)].
    stable_rank = float((s_vals**2).sum() / s_vals[0]**2)

    # Restrict to the top-r singular subspaces; leverage/coherence are properties of THOSE subspaces.
    row_leverage = np.linalg.norm(U[:, :rank], axis=1)**2   # ||u_i||^2, sums to rank
    col_leverage = np.linalg.norm(Vt[:rank], axis=0)**2     # ||v_j||^2, sums to rank
    irs = row_leverage / row_leverage.sum()                 # row i's share of the structure (sums to 1)
    ics = col_leverage / col_leverage.sum()                 # col j's share of the structure (sums to 1)

    # Coherence: mu in [1, dim/rank]. ~1 => structure spread evenly across rows/cols >>1 => a few rows/cols carry it.
    row_coherence = (m/rank) * row_leverage.max()
    col_coherence = (n/rank) * col_leverage.max()

    # Spikiness: ||M||_inf / (||M||_F / sqrt(mn)), max entry vs rms entry. ~1 => energy spread evenly
    # across entries, >>1 => a few entries dominate. Magnitude-aware (unlike coherence) so it bounds
    # entrywise (l_inf) recoverability directly.
    spikiness = np.abs(matrix).max() / (np.linalg.norm(matrix) / np.sqrt(m*n))

    # Sparsity, counting non-zero rows and columns in the matrix
    tol = s_vals[0] * max(m,n) * np.finfo(matrix.dtype).eps
    nnz_rows = np.count_nonzero(np.abs(matrix).sum(axis=1) > tol)
    nnz_cols = np.count_nonzero(np.abs(matrix).sum(axis=0) > tol)

    return (rank, stable_rank, spikiness, (m,n),
            irs, ics, row_coherence, col_coherence,
            nnz_rows, nnz_cols)