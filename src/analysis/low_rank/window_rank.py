"""Loaders and metrics for the penalised-window Hankel spectrum probe.

The in-training probe (agents.sb3_fhr._FHRMixin._fhr_window_rank_probe and
agents.fhrdqn_agent.FHRDQNAgent._window_rank_probe, enabled by the agent
config key window_rank_every > 0) writes, per run directory:

    window_hankel.csv    one row per (probe tick, critic): grad_step,
                         env_steps, window_len, penalty_len, n_windows,
                         unique_eps, critic, sv_NN (stacked window matrix),
                         pen_sv_NN (the trailing penalty sub-window) —
                         nan-padded to a fixed schema
    window_matrices/     the raw (n_windows x window_len) Q window matrices,
                         gsNNNNNNNN_wsNNNNNNNN_cI.npy

This module is the notebook-facing reader: per-tick rank/ratio metrics
(critic-averaged), per-arm aggregation, and a manifest scanner that finds
every instrumented run under an experiment's cached/ directory. Rank
conventions match analysis.low_rank.rank.energy_rank.
"""
import csv
import json
import pathlib

import numpy as np

from analysis.low_rank.rank import energy_rank


def load_window_hankel(run_dir):
    """window_hankel.csv of one run -> list of float-valued row dicts
    (the critic column stays int); [] when the run was not instrumented."""
    path = pathlib.Path(run_dir) / "window_hankel.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for raw in csv.DictReader(f):
            row = {k: float(v) for k, v in raw.items()}
            row["critic"] = int(row["critic"])
            rows.append(row)
    return rows


def tick_metrics(rows):
    """Per-tick metrics, averaged over critics, as a dict of aligned arrays.

    Keys: grad_step, env_steps, n_windows, rank999, rank99 (energy ranks of
    the full window spectrum), s2_s1, s3_s1 (full-window sigma ratios), and
    pen_tail_ratio = sigma_{r+1}/sigma_1 of the (r+1)-column penalty block —
    the relative size of the component an order-r shared recurrence CAN
    remove; an FHR arm that compresses what it penalises drives this down
    faster than the lambda=0 baseline. Ticks with no valid window are
    dropped."""
    sv_keys = sorted(k for k in (rows[0] if rows else {}) if k.startswith("sv_"))
    pen_keys = sorted(k for k in (rows[0] if rows else {})
                      if k.startswith("pen_sv_"))
    by_tick: dict[float, list[dict]] = {}
    for r in rows:
        by_tick.setdefault(r["grad_step"], []).append(r)
    out = {k: [] for k in ("grad_step", "env_steps", "n_windows", "rank999",
                           "rank99", "s2_s1", "s3_s1", "pen_tail_ratio")}
    for gs in sorted(by_tick):
        per_critic = {k: [] for k in ("rank999", "rank99", "s2_s1", "s3_s1",
                                      "pen_tail_ratio")}
        for r in by_tick[gs]:
            svs = np.array([r[k] for k in sv_keys])
            svs = svs[np.isfinite(svs)]
            pens = np.array([r[k] for k in pen_keys])
            pens = pens[np.isfinite(pens)]
            if svs.size == 0 or r["n_windows"] <= 0:
                continue
            per_critic["rank999"].append(energy_rank(svs, 0.999))
            per_critic["rank99"].append(energy_rank(svs, 0.99))
            per_critic["s2_s1"].append(svs[1] / svs[0] if svs.size > 1 else np.nan)
            per_critic["s3_s1"].append(svs[2] / svs[0] if svs.size > 2 else np.nan)
            per_critic["pen_tail_ratio"].append(
                pens[-1] / pens[0] if pens.size > 1 else np.nan)
        if not per_critic["rank999"]:
            continue
        out["grad_step"].append(gs)
        out["env_steps"].append(by_tick[gs][0]["env_steps"])
        out["n_windows"].append(by_tick[gs][0]["n_windows"])
        for k, vals in per_critic.items():
            out[k].append(float(np.nanmean(vals)))
    return {k: np.asarray(v) for k, v in out.items()}


def arm_tick_metrics(run_dirs):
    """[(seed, run_dir), ...] -> [(seed, tick_metrics(...)), ...], skipping
    uninstrumented runs."""
    out = []
    for seed, d in run_dirs:
        rows = load_window_hankel(d)
        if rows:
            out.append((seed, tick_metrics(rows)))
    return out


def final_quarter_summary(metrics_by_seed):
    """Mean of each metric over the final quarter of training (by env_steps),
    pooled across the arm's seeds. -> dict or None when nothing is there."""
    pools: dict[str, list[float]] = {}
    for _, m in metrics_by_seed:
        if m["env_steps"].size == 0:
            continue
        cut = m["env_steps"].max() * 0.75
        sel = m["env_steps"] >= cut
        for k in ("rank999", "rank99", "s2_s1", "s3_s1", "pen_tail_ratio"):
            pools.setdefault(k, []).extend(np.asarray(m[k])[sel].tolist())
    if not pools:
        return None
    return {k: float(np.nanmean(v)) for k, v in pools.items()}


def discover_probe_runs(cached_dir="cached"):
    """Scan every *manifest*.json under cached_dir for runs that carry
    window_hankel.csv. -> {manifest_filename: {arm: [(seed, run_dir), ...]}},
    arms in manifest order, seeds in manifest seed order; manifests (or arms)
    with no instrumented run are omitted."""
    cached = pathlib.Path(cached_dir)
    families: dict[str, dict[str, list]] = {}
    for mpath in sorted(cached.glob("*manifest*.json")):
        try:
            manifest = json.load(open(mpath))
            runs = manifest["runs"]
            seeds = [str(s) for s in manifest.get("seeds", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        arms = {}
        for arm, by_seed in runs.items():
            hits = []
            for s in (seeds or sorted(by_seed)):
                rel = by_seed.get(s)
                if rel is None:
                    continue
                run_dir = cached.parent / rel
                if (run_dir / "window_hankel.csv").exists():
                    hits.append((s, run_dir))
            if hits:
                arms[arm] = hits
        if arms:
            families[mpath.name] = arms
    return families
