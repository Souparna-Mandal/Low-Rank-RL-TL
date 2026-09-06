"""The binned diagnostics cache is the committed reproducibility artefact:
a run directory holding only diag_binned_600.npz (no raw
train_diagnostics.csv) must still feed the figure toolkit."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from analysis.visualisations.fhr_figures import Family  # noqa: E402


def test_load_diag_uses_cache_without_raw_csv(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "diag_binned_600.npz", env_steps=np.arange(5.0),
             td_loss=np.ones(5, np.float32))
    fam = Family.__new__(Family)          # no config/manifest needed here
    fam._diag_cache = {}
    out = fam.load_diag(run, n_bins=600)
    assert out is not None and set(out) == {"env_steps", "td_loss"}
    assert fam.load_diag(run / "missing", n_bins=600) is None
