import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_spectra(matrix: np.ndarray):
    svals = np.linalg.svd(matrix, compute_uv=False)
    
    # bar plot of eigvals and log eigvals side by side
    mags = np.sort(np.abs(svals))[::-1]     # spectrum in descending magnitude
    idx = np.arange(1, len(mags) + 1)
    eps = np.finfo(mags.dtype).eps          # lower limit so log(0) doesn't blow up

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    ax_lin.bar(idx, mags, color="steelblue")
    ax_lin.set_xlabel("index")
    ax_lin.set_ylabel(r"$|\lambda|$")
    ax_lin.set_title("Singular Value magnitudes")

    ax_log.bar(idx, np.log10(mags + eps), color="indianred")
    ax_log.set_xlabel("index")
    ax_log.set_ylabel(r"$\log_{10}|\lambda|$")
    ax_log.set_title("Log Singular Value magnitudes")

    fig.tight_layout()
    plt.show()
    return svals