#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ycf/.conda/envs/torch/bin/python}"
LOG_PATH="${ONPOLICY_STDOUT_LOG:-$ROOT/logs/douzero_4a4_pid_ppo_onpolicy_train.log}"

mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT"

"$PYTHON_BIN" scripts/train_douzero_4a4_distributed.py "$@" 2>&1 | tee -a "$LOG_PATH"
