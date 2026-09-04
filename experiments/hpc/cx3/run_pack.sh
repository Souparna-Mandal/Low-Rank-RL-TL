#!/bin/bash
# Run a PACK of consecutive jobs-file lines concurrently on ONE GPU — the
# per-user cap is 12 running GPUs, so sharing a card is the only way past 12
# concurrent runs (a single effrainbow run leaves roughly half the GPU idle;
# 2 per L40S is the safe default, verify 3 with nvitop before trying it).
#
#   run_pack.sh <jobs.txt> <pack index (1-based)> <pack size>
#
# Pack k covers lines (k-1)*PACK+1 .. k*PACK. Each run's output goes to its
# own file under logs/ next to the jobs file; exit is non-zero if ANY run in
# the pack failed (so --skip-existing resubmission picks the stragglers up).
set -uo pipefail
JOBS_FILE=$1
PACK_INDEX=$2
PACK=$3

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL=$(wc -l < "$JOBS_FILE")
mkdir -p "$HERE/logs"

pids=() lines=()
for k in $(seq 1 "$PACK"); do
    IDX=$(( (PACK_INDEX - 1) * PACK + k ))
    [ "$IDX" -le "$TOTAL" ] || continue
    LOG="$HERE/logs/$(basename "$JOBS_FILE" .txt)_line${IDX}.log"
    bash "$HERE/run_one.sh" "$JOBS_FILE" "$IDX" > "$LOG" 2>&1 &
    pids+=($!); lines+=("$IDX")
    echo "[run_pack] line $IDX started (log: $LOG)"
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[run_pack] line ${lines[$i]} OK"
    else
        echo "[run_pack] line ${lines[$i]} FAILED" >&2
        rc=1
    fi
done
exit $rc
