#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/same_go"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/01_reproduce_same_go.log"
BASELINE_LOG="$LOG_DIR/02_baselines_same_go.log"
SOTA_LOG="$LOG_DIR/03_sota_same_go.log"
RUN_LOG="$LOG_DIR/run_same_go_full.log"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

cleanup_mem() {
  {
    echo "[$(ts)] [CLEANUP] sync + drop_caches + swap recycle (best-effort)"
    sync || true
    if [[ -w /proc/sys/vm/drop_caches ]]; then
      echo 3 > /proc/sys/vm/drop_caches || true
    fi
    if command -v swapoff >/dev/null 2>&1 && command -v swapon >/dev/null 2>&1; then
      swapoff -a || true
      swapon -a || true
    fi
    python - <<'PY'
import gc
gc.collect()
print("python gc.collect() done")
PY
    free -h || true
  } >> "$RUN_LOG" 2>&1
}

echo "[$(ts)] START same_go full pipeline" | tee -a "$RUN_LOG"
echo "[$(ts)] Logs: $LOG_DIR" | tee -a "$RUN_LOG"

cleanup_mem

echo "[$(ts)] [STEP 1/4] Build ProteinPrompt docker image (if needed)" | tee -a "$RUN_LOG"
docker image inspect proteinprompt >/dev/null 2>&1 || docker build -t proteinprompt "$ROOT_DIR/proteinPrompt" >> "$RUN_LOG" 2>&1

cleanup_mem

echo "[$(ts)] [STEP 2/4] Reproduce HybridStackPPI same_go (human+yeast)" | tee -a "$RUN_LOG"
python scripts/reproduce_results.py \
  --dataset both \
  --pairing hadamard_abs \
  --strategy same_go \
  2>&1 | tee "$MAIN_LOG"

cleanup_mem

echo "[$(ts)] [STEP 3/4] Classical baselines same_go (CT/AC)" | tee -a "$RUN_LOG"
python scripts/run_baselines.py \
  --dataset both \
  --strategy same_go \
  2>&1 | tee "$BASELINE_LOG"

cleanup_mem

echo "[$(ts)] [STEP 4/4] SOTA baselines same_go (SPRINT, D-SCRIPT, ESM2-MLP, ProteinPrompt, RaftPPI)" | tee -a "$RUN_LOG"
python scripts/run_sota.py \
  --methods sprint dscript esm2_mlp proteinprompt raftppi \
  --dataset both \
  --strategy same_go \
  2>&1 | tee "$SOTA_LOG"

cleanup_mem

echo "[$(ts)] DONE same_go full pipeline" | tee -a "$RUN_LOG"
