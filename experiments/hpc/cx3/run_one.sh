#!/bin/bash
# Run ONE training job from a jobs file (line = game, config, arm, seed,
# overrides — see make_jobs.py). Called by the PBS array scripts with
# $PBS_ARRAY_INDEX, but works standalone for a login-node smoke test:
#
#   bash experiments/hpc/cx3/run_one.sh experiments/hpc/cx3/jobs_tune.txt 1
#
# Child mode never touches manifests — rebuild them after the sweep with
# experiments/src/rebuild_atari_manifests.py.
set -euo pipefail
JOBS_FILE=$1
INDEX=$2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LINE=$(sed -n "${INDEX}p" "$JOBS_FILE")
[ -n "$LINE" ] || { echo "no line $INDEX in $JOBS_FILE" >&2; exit 1; }
IFS=$'\t' read -r GAME CONFIG ARM_KEY SEED OVERRIDES <<< "$LINE"

echo "[run_one] $GAME $ARM_KEY seed $SEED ($CONFIG)"
cd "$REPO/experiments/atari/$GAME"

ARGS=(--seed "$SEED" --config "$CONFIG")
if [ "$ARM_KEY" = "baseline" ]; then
    ARGS+=(--arm baseline)
else
    ARGS+=(--arm fhr --name-tag "$ARM_KEY" --agent-overrides "$OVERRIDES")
fi
if [ -n "${SMOKE_STEPS:-}" ]; then      # short shakedown run, tiny final eval
    ARGS+=(--steps "$SMOKE_STEPS" --eval-episodes 2)
fi

exec python ../../src/run_fhrdqn_atari100k.py "${ARGS[@]}"
