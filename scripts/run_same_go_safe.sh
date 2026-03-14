#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/same_go"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/01_reproduce_same_go.log"
BASELINE_LOG="$LOG_DIR/02_baselines_same_go.log"
SOTA_LOG="$LOG_DIR/03_sota_same_go.log"
RUN_LOG="$LOG_DIR/run_same_go_full.log"

: > "$RUN_LOG"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  PY_BIN="$(command -v python3)"
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START same_go safe pipeline" | tee -a "$RUN_LOG"

echo "[$(ts)] [STEP 1/4] Preflight checks" | tee -a "$RUN_LOG"
require_file() {
  if [[ ! -f "$1" ]]; then echo "Missing $1"; exit 1; fi
}
require_file "$ROOT_DIR/data/BioGrid/Human/human_pairs_same_go.tsv"
require_file "$ROOT_DIR/data/BioGrid/Yeast/yeast_pairs_same_go.tsv"

echo "[$(ts)] [STEP 2/4] Reproduce HybridStackPPI same_go" | tee -a "$RUN_LOG"
"$PY_BIN" scripts/entrypoints/reproduce_results.py \
  --dataset both \
  --pairing hadamard_abs \
  --strategy same_go \
  2>&1 | tee "$MAIN_LOG"

echo "[$(ts)] [STEP 3/4] Classical baselines same_go" | tee -a "$RUN_LOG"
"$PY_BIN" scripts/entrypoints/run_baselines.py \
  --dataset both \
  --strategy same_go \
  2>&1 | tee "$BASELINE_LOG"

echo "[$(ts)] [STEP 4/4] SOTA baselines same_go (Skipping Docker if missing)" | tee -a "$RUN_LOG"

# Identify which methods to run based on Docker availability
METHODS="sprint dscript esm2_mlp"
if docker image inspect proteinprompt >/dev/null 2>&1; then
  METHODS="$METHODS proteinprompt"
fi
if docker image inspect raftppi >/dev/null 2>&1; then
  METHODS="$METHODS raftppi"
fi

echo "Running SOTA methods: $METHODS" | tee -a "$SOTA_LOG"
"$PY_BIN" scripts/entrypoints/run_sota.py \
  --methods $METHODS \
  --dataset both \
  --strategy same_go \
  2>&1 | tee -a "$SOTA_LOG"

echo "[$(ts)] DONE same_go safe pipeline" | tee -a "$RUN_LOG"
