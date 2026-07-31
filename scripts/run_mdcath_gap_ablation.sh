#!/usr/bin/env bash
set -uo pipefail

RUN_PYTHON="${RUN_PYTHON:-/opt/conda/envs/acvla/bin/python}"
RUN_LOG_DIR="${RUN_LOG_DIR:-ablation_logs}"

mkdir -p "$RUN_LOG_DIR"

for gap in 1 2 5; do
  checkpoint_dir="checkpoints_mdcath_gap${gap}"
  log_path="$RUN_LOG_DIR/gap${gap}.log"
  if [[ -e "$checkpoint_dir" || -e "$log_path" ]]; then
    echo "Refusing to overwrite existing output: $checkpoint_dir or $log_path" >&2
    exit 2
  fi
done

run_gap() {
  local gap="$1"
  local gpu="$2"
  local log_path="$RUN_LOG_DIR/gap${gap}.log"
  echo "[$(date --iso-8601=seconds)] starting gap=$gap on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$RUN_PYTHON" train.py \
    --config "configs/ablations/mdcath_gap${gap}.yaml" 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  echo "[$(date --iso-8601=seconds)] finished gap=$gap status=$status"
  return "$status"
}

run_gap 1 0 &
gap1_pid="$!"
run_gap 2 1 &
gap2_pid="$!"

overall_status=0
if wait "$gap1_pid"; then
  run_gap 5 0 || overall_status=1
else
  echo "gap=1 failed; gap=5 was not started" >&2
  overall_status=1
fi

wait "$gap2_pid" || overall_status=1
exit "$overall_status"
