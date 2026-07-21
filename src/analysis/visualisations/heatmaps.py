import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_heatmap(matrix: np.ndarray, matrix_name: str,
                        save_to=None, show: bool = True):
    """Heatmap of the matrix. Colour scale spans the matrix's own min/max for full contrast."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_title(f"Heatmap of {matrix_name}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
