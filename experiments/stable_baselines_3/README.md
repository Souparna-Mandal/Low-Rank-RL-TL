# SB3 (Stable-Baselines3) FHR comparison experiments

Stable-Baselines3's best published sample-efficient DQN-family method —
**RL-Zoo-tuned QR-DQN** from `sb3_contrib` (or plain SB3 `DQN` via
`algo.type: dqn`; the zoo tuning is identical apart from `n_quantiles`) —
versus the same method with the **FHR learned-recurrence (Hankel-rank)
penalty** bootstrapped onto its TD loss. The penalty, warm-up, ramp-down and
diagnostics semantics are 1:1 with the classic `FHRDQNAgent`
(`src/agents/fhrdqn_agent.py`); the SB3 subclasses live in
`src/agents/sb3_fhr.py`, and the **baseline arm (`fhr_weight: 0.0`) is
bit-for-bit the stock SB3 algorithm** — same RNG stream, same updates —
asserted in `tests/test_sb3_fhr.py`. An `exp<N>`-vs-baseline gap is therefore
attributable to the FHR penalty alone.

## Layout

The shared SB3 runners plus one experiment dir per env, each with the
standard repo structure:

    experiments/stable_baselines_3/
      src/
        run_sb3_seeds.py      multi-seed launcher (launch_all/load_runs/videos)
        run_sb3_analysis.py   post-hoc run-every-analysis-framework CLI
      cartpole/      CartPole-v1     configs/config_sb3.yaml + cached/ + notebooks
      mountaincar/   MountainCar-v0  "
      acrobot/       Acrobot-v1      "
      pendulum/      Pendulum-v1     SAC track: configs/config_sb3_sac.yaml +
      fetch_reach/   FetchReach      cached/ + exp1_fhrsac_results.ipynb
      halfcheetah/   HalfCheetah-v5  MuJoCo locomotion, RL-Zoo mujoco-defaults
      ant/           Ant-v5          "
      swimmer/       Swimmer-v5      "
      humanoid_standup/ HumanoidStandup-v5  "
      humanoid/      Humanoid-v5     "  (regular locomotion Humanoid, 1e6 steps)

The result viewer scans `experiments/` only, and this suite sits directly
under it, so the SB3 runs appear in the app under the exp keys
`stable_baselines_3/<env>` with zero viewer configuration. (Until 2026-08-25
the tree lived at the repo root and `experiments/stable_baselines_3` was a
symlink to it; the move left those exp keys unchanged.)

Per experiment dir:

- `configs/config_sb3.yaml` — RL-Zoo algo hyperparameters (`algo:`), FHR
  defaults (`agent:`), early-stop protocol (`training:`), and the low-rank
  diagnostics (`analysis:` — Q-matrix rank grid, Hankel sweep, AR value
  probe, all dispatched through the same `training.run_analysis_tick` the
  classic runs use). Numbered variants under `experiment.fhr_experiments`.
- `cached/runs/<name>_<timestamp>/` — the standard per-run artifact contract
  (`rewards.csv` with `episode,reward,steps`, `rank_stats.csv`,
  `hankel_sweep.csv`, `train_diagnostics.csv`, `figures/`, `trajectories/`,
  `autoregressive_*` probe artifacts, `checkpoints/{latest,best,final}.pt` as
  SB3 zip archives). Fully readable by the result viewer app
  (`python result_viewer_app/rank_viewer.py`).
- `cached/sb3_runs_manifest.json` — arm/seed → run-dir manifest (family
  `sb3_runs` in the viewer's compare mode).
- `exp1_sb3_fhrdqn_compare.ipynb` — the comparison notebook (launch cell +
  seed-averaged figures: learning curves on the env-steps axis, TD error,
  penalty, coefficients, recurrence health, penalty batch composition,
  Hankel-rank evolution, Q-matrix rank, summary, videos).
- `exp2_sb3_recurrence.ipynb` — the AR value-recurrence analysis suite over a
  chosen run.

## Running

    cd experiments/stable_baselines_3/cartpole
    python ../src/run_sb3_seeds.py --experiment 1 2   # baseline + exp1 + exp2, all seeds
    python ../src/run_sb3_analysis.py --all           # re-run every analysis framework
                                                      # post-hoc on the final checkpoints

or execute the launch cell of `exp1_sb3_fhrdqn_compare.ipynb`. The launcher
skips (arm, seed) pairs already in the manifest; `--force` reruns them
(the shared baseline is only retrained by `--force` without `--experiment`).
Child logs: `cached/logs/sb3_<arm>_seed<N>.log`.
