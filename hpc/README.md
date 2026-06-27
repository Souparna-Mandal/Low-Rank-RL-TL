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

## Personal Note

PBS is the job scheduler used when we specify the `qsub` command. We put those as comments in our bash file we submit. 

```bash
-N <job-name>
```

`#PBS -lselect=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=A100` is for the resource request. Breaking down the colon-separated chunks:

select=1 — 1 node
ncpus=4 — 4 CPU cores on that node
mem=16gb — 16 GB RAM
ngpus=1 — 1 GPU
gpu_type=A100 — specifically an A100 (omit this to get any available GPU)