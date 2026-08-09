import csv
import pathlib
import shutil
from datetime import datetime

import numpy as np


class RunLogger:
    """Collects everything one training run produces under
    <exp_dir>/runs/<run_id>/:

        config.yaml     copy of the config the run was launched with
        rewards.csv     episode, reward — rewritten at every analysis tick and at the end.
        rank_stats.csv  one row per (episode, matrix) from row_rank_property_check,
                        so rank-vs-training-progress can be plotted after the fact this can be any statistic that's tracked and not just rank
        figures/        epNNNNNN_<matrix>.png spectra during training, final_* after
        checkpoints/    latest.pt (rolling), best.pt (best reward window), final.pt

    Pass an instance to `dqn_training_loop(run_logger=...)`; everything else is
    Optional convenience for notebook cells (figure_path, checkpoint).
    """

    def __init__(self, exp_dir, config_path=None, run_id: str | None = None):
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = pathlib.Path(exp_dir) / "runs" / run_id
        self.figures_dir = self.dir / "figures"
        self.checkpoints_dir = self.dir / "checkpoints"
        self.trajectories_dir = self.dir / "trajectories"
        self.mc_rollouts_dir = self.dir / "mc_rollouts"
        self.autoregressive_rollouts_dir = self.dir / "autoregressive_rollouts"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.trajectories_dir.mkdir(exist_ok=True)
        self.mc_rollouts_dir.mkdir(exist_ok=True)
        if config_path is not None:
            shutil.copy(config_path, self.dir / "config.yaml")
        self._rank_csv = self.dir / "rank_stats.csv"
        self._sweep_csv = self.dir / "hankel_sweep.csv"

    @staticmethod
    def _slug(name: str) -> str:
        return "-".join("".join(c if c.isalnum() else " " for c in name.lower()).split())

    def figure_path(self, name: str, episode: int | None = None) -> pathlib.Path:
        """Namespaced figure file: ep000100_hankel-q.png during training,
        final_hankel-q.png (episode=None) for post-training figures."""
        tag = "final" if episode is None else f"ep{episode:06d}"
        return self.figures_dir / f"{tag}_{self._slug(name)}.png"

    def log_rank_stats(self, episode, matrix_name, r, sr, spk, shape,
                       irs, ics, rc, cc, nzr, nzc) -> None:
        """Append one row of row_rank_property_check results for a matrix."""
        header_needed = not self._rank_csv.exists()
        with open(self._rank_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["episode", "matrix", "eff_rank", "stable_rank",
                                 "spikiness", "n_rows", "n_cols", "nnz_rows", "nnz_cols",
                                 "row_coherence", "col_coherence",
                                 "row_lev_min", "row_lev_max", "col_lev_min", "col_lev_max"])
            writer.writerow([episode, matrix_name, r, f"{sr:.4f}", f"{spk:.4f}",
                             shape[0], shape[1], nzr, nzc,
                             f"{rc:.4f}", f"{cc:.4f}",
                             f"{irs.min():.6g}", f"{irs.max():.6g}",
                             f"{ics.min():.6g}", f"{ics.max():.6g}"])

    def log_hankel_sweep(self, episode, matrix_name, rollout_idx, seed, sub_len,
                         r, sr, spk, shape, irs, ics, rc, cc, nzr, nzc) -> None:
        """Append one row of the multi-rollout / sub-trajectory Hankel sweep.
        Same metric columns as rank_stats.csv, prefixed with rollout/seed/sub_len
        so a metrics-vs-(sub_len) curve can be reconstructed per rollout."""
        header_needed = not self._sweep_csv.exists()
        with open(self._sweep_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["episode", "matrix", "rollout", "seed", "sub_len",
                                 "eff_rank", "stable_rank", "spikiness", "n_rows", "n_cols",
                                 "nnz_rows", "nnz_cols", "row_coherence", "col_coherence",
                                 "row_lev_min", "row_lev_max", "col_lev_min", "col_lev_max"])
            writer.writerow([episode, matrix_name, rollout_idx, seed, sub_len,
                             r, f"{sr:.4f}", f"{spk:.4f}", shape[0], shape[1],
                             nzr, nzc, f"{rc:.4f}", f"{cc:.4f}",
                             f"{irs.min():.6g}", f"{irs.max():.6g}",
                             f"{ics.min():.6g}", f"{ics.max():.6g}"])

    def save_trajectory(self, episode, seed, seqs: dict) -> pathlib.Path:
        """Persist the raw per-step scalar sequences of one rollout as a compressed
        .npz (keys slugified), so Hankel analysis can be recomputed offline."""
        tag = "final" if episode is None else f"ep{episode:06d}"
        path = self.trajectories_dir / f"{tag}_seed{seed}.npz"
        np.savez_compressed(path, **{self._slug(name): arr for name, arr in seqs.items()})
        return path

    def save_mc_rollouts(self, episode, rollouts: dict) -> pathlib.Path:
        """Persist one PI iteration's generative Monte-Carlo evaluation rollouts
        (flat arrays keyed by name) as a compressed .npz, so truncated/low-rank
        evaluation can be studied offline. One file per iteration."""
        tag = "final" if episode is None else f"ep{episode:06d}"
        path = self.mc_rollouts_dir / f"{tag}.npz"
        np.savez_compressed(path, **rollouts)
        return path

    def log_train_diagnostics(self, episode, **metrics) -> None:
        """Append one row of agent.train() diagnostics (td_loss, penalty terms,
        rank/gate stats, ...) to train_diagnostics.csv. Columns come from the
        metric keys on first write, so any agent's diagnostics dict fits."""
        path = self.dir / "train_diagnostics.csv"
        header_needed = not path.exists()
        keys = sorted(metrics)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["episode"] + keys)
            writer.writerow([episode] + [f"{metrics[k]:.6g}" for k in keys])

    # -- autoregressive value-recurrence probe ------------------------------
    AUTOREGRESSIVE_METRIC_COLUMNS = [
        "episode", "order", "split", "subset",
        "rmse_one_step_ahead", "one_minus_r_squared_one_step_ahead",
        "n_sequences_scored",
        "n_trajectories_collected", "mean_trajectory_length",
    ]
    AUTOREGRESSIVE_COEFFICIENT_COLUMNS = ["episode", "order", "split", "lag",
                                          "coefficient"]

    def log_autoregressive_metrics(self, rows) -> None:
        """Append one-step-ahead fit rows to autoregressive_value_metrics.csv.

        One row per (episode, recurrence order, split, training/test subset),
        as produced by analysis.low_rank.autoregressive_value_probe.metric_rows.
        Errors at longer forecast windows go through
        log_autoregressive_horizon_metrics instead, one row per window.
        """
        self._append_rows(self.dir / "autoregressive_value_metrics.csv",
                          self.AUTOREGRESSIVE_METRIC_COLUMNS, rows)

    AUTOREGRESSIVE_HORIZON_COLUMNS = [
        "episode", "order", "split", "subset", "forecast_horizon",
        "rmse", "one_minus_r_squared", "diverged", "mean_trajectory_length",
    ]

    def log_autoregressive_horizon_metrics(self, rows) -> None:
        """Append the rolling-horizon sweep to
        autoregressive_value_horizon_metrics.csv.

        One row per (episode, order, split, subset, forecast horizon), where the
        horizon is how many steps are forecast before the recurrence is
        re-anchored on the values that actually occurred.
        """
        self._append_rows(
            self.dir / "autoregressive_value_horizon_metrics.csv",
            self.AUTOREGRESSIVE_HORIZON_COLUMNS, rows)

    def log_autoregressive_coefficients(self, rows) -> None:
        """Append fitted coefficients to autoregressive_value_coefficients.csv.

        One row per (episode, order, split, lag), so each order's coefficient
        trajectory over training plots directly. lag 0 is the intercept.
        """
        self._append_rows(self.dir / "autoregressive_value_coefficients.csv",
                          self.AUTOREGRESSIVE_COEFFICIENT_COLUMNS, rows)

    def save_autoregressive_example_rollouts(self, episode, arrays) -> pathlib.Path:
        """Persist actual-vs-predicted example sequences for one checkpoint.

        Saved under autoregressive_rollouts/epNNNNNN.npz so the notebook and the
        result viewer can plot the same arrays without recomputing them.
        """
        self.autoregressive_rollouts_dir.mkdir(exist_ok=True)
        tag = "final" if episode is None else f"ep{episode:06d}"
        path = self.autoregressive_rollouts_dir / f"{tag}.npz"
        np.savez_compressed(path, **arrays)
        return path

    @staticmethod
    def _append_rows(path, columns, rows) -> None:
        """Append dict rows to a CSV, writing the header on first use."""
        if not rows:
            return
        header_needed = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if header_needed:
                writer.writeheader()
            writer.writerows(rows)

    def log_rewards(self, rewards) -> None:
        with open(self.dir / "rewards.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            writer.writerows(enumerate(rewards))

    def checkpoint(self, agent, name: str = "latest") -> pathlib.Path:
        path = self.checkpoints_dir / f"{name}.pt"
        agent.save(path)
        return path
