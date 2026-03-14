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

: > "$RUN_LOG"
: > "$MAIN_LOG"
: > "$BASELINE_LOG"
: > "$SOTA_LOG"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x "/media/ssd/conda/envs/ppis_env/bin/python" ]]; then
  PY_BIN="/media/ssd/conda/envs/ppis_env/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  PY_BIN="$(command -v python3)"
fi

if [[ -n "${HF_HOME:-}" ]]; then
  RESOLVED_HF_HOME="$HF_HOME"
elif [[ -d "$HOME/.cache/huggingface" ]]; then
  RESOLVED_HF_HOME="$HOME/.cache/huggingface"
elif [[ -d "/media/SAS/Van/ppis/.hf_cache" ]]; then
  RESOLVED_HF_HOME="/media/SAS/Van/ppis/.hf_cache"
else
  RESOLVED_HF_HOME=""
fi

if [[ -n "$RESOLVED_HF_HOME" ]]; then
  export HF_HOME="$RESOLVED_HF_HOME"
  export TRANSFORMERS_CACHE="$RESOLVED_HF_HOME/hub"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ts() { date +"%Y-%m-%d %H:%M:%S"; }

cleanup_mem() {
  {
    echo "[$(ts)] [CLEANUP] sync + drop_caches + swap recycle (best-effort)"
    sync || true
    if [[ -w /proc/sys/vm/drop_caches ]]; then
      echo 3 > /proc/sys/vm/drop_caches || true
    else
      echo "[$(ts)] [CLEANUP] drop_caches skipped (needs root write on /proc/sys/vm/drop_caches)"
    fi
    if command -v swapoff >/dev/null 2>&1 && command -v swapon >/dev/null 2>&1; then
      swapoff -a || true
      swapon -a || true
    else
      echo "[$(ts)] [CLEANUP] swap recycle skipped (swapoff/swapon not available)"
    fi
    "$PY_BIN" - <<'PY'
import gc
gc.collect()
print("python gc.collect() done")
PY
    free -h || true
  } >> "$RUN_LOG" 2>&1
}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[$(ts)] [PREFLIGHT][ERROR] Missing required file: $f" | tee -a "$RUN_LOG"
    exit 1
  fi
}

check_raft_cache() {
  "$PY_BIN" - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", local_files_only=True)
print("raft_esm2_t6_8M_UR50D: OK")
PY
}

check_hybridstack_esm2_cache() {
  "$PY_BIN" - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D", local_files_only=True)
print("hybridstack_esm2_t33_650M_UR50D: OK")
PY
}

echo "[$(ts)] START same_go full pipeline" | tee -a "$RUN_LOG"
echo "[$(ts)] Logs: $LOG_DIR" | tee -a "$RUN_LOG"
echo "[$(ts)] Python: $PY_BIN" | tee -a "$RUN_LOG"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "[$(ts)] HF_HOME: $HF_HOME" | tee -a "$RUN_LOG"
fi

cleanup_mem

echo "[$(ts)] [STEP 1/4] Preflight checks (dataset, docker image, model caches)" | tee -a "$RUN_LOG"
require_file "$ROOT_DIR/data/BioGrid/Human/human_pairs_same_go.tsv"
require_file "$ROOT_DIR/data/BioGrid/Yeast/yeast_pairs_same_go.tsv"
require_file "$ROOT_DIR/cache/esm2/esm2_embeddings_v4.h5"

if ! docker image inspect proteinprompt >/dev/null 2>&1; then
  echo "[$(ts)] [PREFLIGHT][ERROR] Docker image 'proteinprompt' not found. Build it first." | tee -a "$RUN_LOG"
  exit 1
fi

if ! check_raft_cache >> "$RUN_LOG" 2>&1; then
  echo "[$(ts)] [PREFLIGHT][ERROR] Missing local cache for facebook/esm2_t6_8M_UR50D (RaftPPI)." | tee -a "$RUN_LOG"
  exit 1
fi

if ! check_hybridstack_esm2_cache >> "$RUN_LOG" 2>&1; then
  echo "[$(ts)] [PREFLIGHT][WARN] Local HF cache for facebook/esm2_t33_650M_UR50D not found." | tee -a "$RUN_LOG"
  echo "[$(ts)] [PREFLIGHT][WARN] HybridStackPPI will run in cached-only mode from esm2_embeddings_v4.h5." | tee -a "$RUN_LOG"
fi

cleanup_mem

echo "[$(ts)] [STEP 2/4] Reproduce HybridStackPPI same_go (human+yeast)" | tee -a "$RUN_LOG"
"$PY_BIN" scripts/entrypoints/reproduce_results.py \
  --dataset both \
  --pairing hadamard_abs \
  --strategy same_go \
  2>&1 | tee "$MAIN_LOG"

cleanup_mem

echo "[$(ts)] [STEP 3/4] Classical baselines same_go (CT/AC)" | tee -a "$RUN_LOG"
"$PY_BIN" scripts/entrypoints/run_baselines.py \
  --dataset both \
  --strategy same_go \
  2>&1 | tee "$BASELINE_LOG"

cleanup_mem

echo "[$(ts)] [STEP 4/4] SOTA baselines same_go (SPRINT, D-SCRIPT, ESM2-MLP, ProteinPrompt, RaftPPI)" | tee -a "$RUN_LOG"
"$PY_BIN" scripts/entrypoints/run_sota.py \
  --methods sprint dscript esm2_mlp proteinprompt raftppi \
  --dataset both \
  --strategy same_go \
  2>&1 | tee "$SOTA_LOG"

cleanup_mem

echo "[$(ts)] DONE same_go full pipeline" | tee -a "$RUN_LOG"
