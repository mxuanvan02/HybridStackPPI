#!/bin/bash
# HybridStack-PPI Full Experimental Pipeline (Auto-Routing Enabled)
# Runs all CV and Ablation experiments sequentially
# Created: 2024-12-28

set -e  # Exit on error

# Ensure logs directory exists
mkdir -p logs

echo "========================================"
echo "  HybridStack-PPI Full Experiment Suite"
echo "  Mode: Auto-Routing + Strict C3 Split"
echo "  Started: $(date)"
echo "========================================"

cd /media/SAS/Van/HybridStackPPI

# 1. Yeast C3 CV
echo ""
echo "[1/4] Yeast C3 Cross-Validation..."
python -u scripts/run_cv.py --dataset yeast --n-splits 5 --n-jobs 1 2>&1 | tee logs/yeast_strict_cv.log
echo "  ✅ Yeast CV Complete!"

# 2. Human C3 CV
echo ""
echo "[2/4] Human C3 Cross-Validation..."
python -u scripts/run_cv.py --dataset human --n-splits 5 --n-jobs 1 2>&1 | tee logs/human_strict_cv.log
echo "  ✅ Human CV Complete!"

# 3. Yeast Ablation
echo ""
echo "[3/4] Yeast Ablation Study..."
python -u scripts/run_full_ablation.py --dataset yeast --n-splits 5 --n-jobs 1 2>&1 | tee logs/yeast_ablation.log
echo "  ✅ Yeast Ablation Complete!"

# 4. Human Ablation
echo ""
echo "[4/4] Human Ablation Study..."
python -u scripts/run_full_ablation.py --dataset human --n-splits 5 --n-jobs 1 2>&1 | tee logs/human_ablation.log
echo "  ✅ Human Ablation Complete!"

echo ""
echo "========================================"
echo "  All Experiments Complete!"
echo "  Finished: $(date)"
echo "========================================"
