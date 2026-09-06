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

## Continuous-action families: SAC and TD3

The same launcher, callback, manifest and notebook contract also carries the
two continuous-action hosts:

- **`algo.type: sac`** — `FHRSAC` (`src/agents/sb3_sac_fhr.py`): SB3 SAC with
  the FHR penalty on both critics, one shared `c` (or `c(s,a)` predictor),
  combined as the per-critic **mean** against SAC's `0.5 * sum_i MSE_i` TD term.
- **`algo.type: td3`** — `FHRTD3` (`src/agents/sb3_td3_fhr.py`): SB3 TD3 with
  the penalty on both critics on **TD3's own scale**: stock TD3's critic loss
  is the plain `sum_i MSE_i`, so the loss is
  `sum_i MSE_i + lambda * sum_i Huber_i` — per critic `MSE_i + lambda*Huber_i`,
  no division by the number of critics. `lambda` is therefore a TD3-scale
  knob, not numerically comparable to the SAC families' `lambda`; the
  cross-algorithm quantities are the stream ratios below. TD3 configs carry
  RL-Zoo's `noise_type` / `noise_std` in the `algo:` block (the launcher turns
  them into SB3's `ActionNoise`), `policy_delay`, `target_policy_noise`,
  `target_noise_clip`, and must not carry `ent_coef` /
  `target_update_interval`. Why TD3: a deterministic actor, a fixed
  exploration noise and a plain Bellman target (no entropy term inside the
  value the recurrence is fitted to) make it the lower-variance host for the
  FHR claim and make the frozen-theory control `c = (1+1/gamma, -1/gamma)` an
  exact identity test.

Both hosts, and the DQN family, share two in-training probes (both run on
every arm including the `lambda = 0` baseline, consume no RNG and leave the
baseline bit-for-bit stock):

- `agent.window_rank_every` — the penalised-window Hankel spectrum
  (`window_hankel.csv`): rank measured on the replay windows the penalty acts
  on;
- `agent.grad_probe_every` — the **gradient-stream probe**: on the same batch,
  `grad_ratio = |d penalty/d theta| / |d TD/d theta|` on the critic
  parameters (unweighted), `grad_rho = lambda_eff * grad_ratio`, `grad_cos`
  (alignment; negative = the streams conflict), next to the loss-side
  `loss_ratio` / `rho_loss`, all in `train_diagnostics.csv`. On the baseline
  the penalty never enters the loss, so its ratio is the free calibration
  signal `lambda* = target / grad_ratio` (`calibrate_fhr.py --by grad`, or
  `Family.table_rho_streams()` in a notebook).

Results notebooks are generated from one template —
`python ../src/make_results_notebook.py --config configs/config_sb3_td3.yaml`
from the experiment dir — and save **every figure individually** under
`figures/<family>/` (training / eval per arm, sample-efficiency ladder,
final returns, one figure per diagnostic, the loss- and gradient-stream rho
figures + selection table, rollout Hankel per signal, window-rank per arm
per metric, cost, videos). The TD3 families live in `pendulum`,
`mountaincar_continuous`, `ant`, `halfcheetah`, `swimmer`
(`configs/config_sb3_td3.yaml`, manifest `cached/sb3_runs_manifest_td3.json`,
notebook `exp1_fhrtd3_results.ipynb`). Waves are launched detached, one
launcher per family:

    cd experiments/stable_baselines_3/ant
    nohup env FHR_CHILD_THREADS=1 ../../../.venv/bin/python ../src/run_sb3_seeds.py \
        --config configs/config_sb3_td3.yaml --experiment 1 2 3 4 5 6 7 \
        --max-workers 20 > ~/localfiles/ant_td3.log 2>&1 & disown

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
