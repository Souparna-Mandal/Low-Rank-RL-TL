# HPC Setup (Imperial College CX3)

## One-time environment setup

Run this interactively on a login node:

```bash
module load Miniforge3/24.11.3-0

conda create -n low-rank-rl python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate low-rank-rl

# Install PyTorch 2.2+ with CUDA 12.1 (matches cluster CUDA/12.1.1)
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121

# Install remaining project dependencies from pyproject.toml
pip install -e /rds/general/user/sm5125/home/Low-Rank-RL-TL
```

## Submitting a job

```bash
cd /rds/general/user/sm5125/home/Low-Rank-RL-TL/hpc
qsub job.pbs
```

## Notes

- The conda env is persistent — you only need to create it once.
- Always load `Miniforge3/24.11.3-0` before activating the env in job scripts.
- Request `ngpus=1` for any training run; omit it for CPU-only debugging.
- Output logs go to `job.pbs.o<jobid>` and `job.pbs.e<jobid>` in the submission directory.

## Monitoring Jobs

```bash
qsub job.pbs              # submit — prints your job ID e.g. 1234567.pbs
qstat                     # all your jobs + status (Q=queued, R=running, E=exiting)
qstat -f 1234567.pbs      # full details on a specific job
qstat -T 1234567.pbs      # estimated start time
qdel 1234567.pbs          # cancel a job
```

Watch logs while running:

```bash
tail -f job.pbs.o         # live stdout
tail -f job.pbs.e         # live stderr
```

Status codes: `Q` = queued, `R` = running, `E` = exiting. Job disappears from `qstat` once complete — check log files at that point.

## PBS Directives Reference

PBS is the job scheduler invoked via `qsub`. `#PBS` lines look like comments to bash but the scheduler reads them before the script runs.

| Directive | Meaning |
|-----------|---------|
| `#PBS -N <job-name>` | Job name shown in `qstat` output |
| `#PBS -lwalltime=HH:MM:SS` | Time limit — job is killed if it exceeds this; shorter walltime tends to get scheduled faster |
| `#PBS -o job.pbs.o` | Stdout log file (written in `$PBS_O_WORKDIR`) |
| `#PBS -e job.pbs.e` | Stderr log file (written in `$PBS_O_WORKDIR`) |

### Resource request (`-lselect`)

```
#PBS -lselect=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=A100
```

| Chunk | Meaning |
|-------|---------|
| `select=1` | Number of nodes |
| `ncpus=4` | CPU cores per node |
| `mem=16gb` | RAM per node |
| `ngpus=1` | GPUs per node |
| `gpu_type=A100` | Specific GPU type (omit to get any available GPU) |