# CX3 (Imperial RCS) — EfficientRainbow ± FHR Atari-100k campaign

Two PBS array campaigns, both driven by the game configs the sync script
renders from `experiments/atari/config_effrainbow_100k.global.yaml`:

| campaign | jobs | budget | arms | seeds |
|---|---|---|---|---|
| **tune** (`tune.pbs`) | 5 games × 8 arms × 2 seeds = **80** | 50k steps | baseline + 7-arm grid around λ2/r8 | 0–1 |
| **suite** (`suite.pbs`) | 27 games × 2 arms × 5 seeds = **270** | 100k steps | baseline + exp3 (λ2, r8) | 0–4 |

Tuning subset (picked from the 1-seed suite's per-game ΔHNS): Boxing (+1.47)
and BankHeist (+0.62) as wins, KungFuMaster and BattleZone as washes,
DemonAttack (−0.33) as the worst loss.

**Where to edit arms**: everything is defined in ONE file,
`experiments/atari/config_effrainbow_100k.global.yaml` — the suite arm under
`fhr_experiments`, the tuning grid under `tune_fhr_experiments` (add/remove
entries there; new numbers ≥ 10). The 5-game subset is `TUNE_GAMES` in
`experiments/src/sync_effrainbow_configs_from_global.py`. After ANY change:

```bash
python experiments/src/sync_effrainbow_configs_from_global.py   # re-render configs
python experiments/hpc/cx3/make_jobs.py --mode tune             # re-list jobs
```

(`make_jobs.py` itself reads the RENDERED per-game configs —
`dqn_<game>/config_effrainbow_tune.yaml` / `config_effrainbow_100k.yaml` —
never the global directly, so always re-sync before re-listing.)

## One-time setup (login node)

```bash
# 1. conda via the RCS-recommended miniforge route (skip if you have it)
module load miniforge/3
miniforge-setup                      # installs ~/miniforge3; then re-login
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda create -n lowrank python=3.12 -y
conda activate lowrank

# 2. repo + deps — compute nodes have NO guaranteed internet: install on the
# login node. The launcher adds src/ to sys.path itself, no editable install.
cd $HOME && git clone <your-remote> Low-Rank-RL-TL
cd Low-Rank-RL-TL && git checkout soup/atari-cx3-hpc
pip install uv && uv pip install -r pyproject.toml
# (mujoco/gymnasium-robotics in there are unused by Atari — fine if they
#  complain, only torch/gymnasium[atari]/ale-py/numpy/yaml/matplotlib/scipy/
#  opencv/tqdm matter here)

# 3. smoke test ON A GPU NODE before any array (checks CUDA wheel + ALE ROMs):
qsub -I -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S -l walltime=0:30:0
conda activate lowrank && cd $HOME/Low-Rank-RL-TL/experiments/atari/dqn_boxing
python ../../src/run_fhrdqn_atari100k.py --arm baseline --seed 0 \
    --config config_effrainbow_tune.yaml --steps 2000 --eval-episodes 2
```

## Submitting

```bash
cd $HOME/Low-Rank-RL-TL
python experiments/hpc/cx3/make_jobs.py --mode tune      # prints the qsub line
mkdir -p experiments/hpc/cx3/logs && cd experiments/hpc/cx3/logs
qsub -J 1-40%12 ../tune.pbs          # 80 runs, PACK=2 per GPU -> 40 subjobs
```

Same flow for the suite with `--mode suite` and `../suite.pbs`
(`qsub -J 1-135%12`, 270 runs). Notes:

* **Parallelism ceiling**: RCS hard-caps every user at **12 running GPUs**,
  so `%12` is the most concurrency an array can get — going wider would just
  queue-block. The extra throughput comes from **PACK=2: each subjob runs 2
  jobs-file lines concurrently on its one GPU** (a single effrainbow run
  leaves roughly half the card idle on our profiling; the L40S has headroom
  to spare) → 24 concurrent runs inside the 12-GPU cap. `qsub -v PACK=1 ...`
  restores one-run-per-GPU (halve nothing else; array must then span all N
  lines); before trying `PACK=3`, watch a PACK=2 subjob with `nvitop`
  (`module load nvitop/...`) and only go wider if GPU util sits well under
  90 %. GPU jobs auto-route to the gpu72 queue (72 h max); never pass `-q`.
* Requests: 1× L40S (48 GB VRAM, the default/plentiful card — don't request
  the scarce A100s), **8 cores + 96 GB RAM per subjob** (the natural
  64-core/8-GPU node ratio; ~6 GB uint8 replay buffer + torch headroom per
  packed run). Walltime 6 h (tune) / 12 h (suite) per PACK is deliberately
  fat: a solo GB10 run of the 100k recipe took ~35–60 min and two packed
  runs contend, so budget ~2× solo.
* Per-run stdout goes to `experiments/hpc/cx3/logs/jobs_<mode>_line<N>.log`
  (live-tailable, unlike the PBS `.o` files which appear only at subjob
  end).
* Submit **from the logs dir** so the array's `.o`/`.e` files collect there
  (PBS writes them to the submission dir, only after each subjob ends).
* Monitor: `qstat -u $USER` (array shows one row, state B), `qstat -t` for
  per-subjob states, `qdel <jobid>` to kill the array.
* Repo somewhere else / different env name?
  `qsub -v REPO=/path,CONDA_ENV=name -J ... ../tune.pbs`.

## After a pass

```bash
# resubmit only what's missing (crashed/timed-out subjobs):
python experiments/hpc/cx3/make_jobs.py --mode suite --skip-existing
cd experiments/hpc/cx3/logs && qsub -J 1-<packs printed above>%12 ../suite.pbs

# child-mode runs never write manifests — rebuild them, then sync home:
python experiments/src/rebuild_atari_manifests.py --family tune
python experiments/src/rebuild_atari_manifests.py                # 100k family
```

Pull results back to your machine (run dirs + manifests; drop `checkpoints/`
if you only need curves and eval numbers):

```bash
rsync -avz --include='*/' --include='cached/runs/**' --include='cached/*.json' \
      --exclude='*' cx3:Low-Rank-RL-TL/experiments/atari/ experiments/atari/
```

Everything lives in `$HOME` (930 GB quota, backed up); a full suite pass is
roughly 30–50 GB of run artifacts. `$EPHEMERAL` auto-deletes after 30 days —
don't leave results there.

## Protocol notes

* The per-game configs now carry env-step-cadenced instrumentation: greedy
  eval checkpoints every 10k env steps (8 episodes → `eval.csv`) and
  rank/Hankel analysis every 10k env steps (`analysis.step_freq`) — episodes
  vary wildly across games, env steps are the protocol constant. Checkpoint
  rollouts consume action-RNG, so results are stream-comparable only among
  runs sharing these settings (i.e. don't mix with the pre-instrumentation
  1-seed suite runs at the RNG level; score-level comparisons are fine).
* Final evaluation is unchanged: 32 full-game episodes at ε = 0.001.
* Tuning runs live in their own manifest family (`effrainbowtune`, run names
  `*_effrainbowtune_*`) and can never pool with the 100k family.
