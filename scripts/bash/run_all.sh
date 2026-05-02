#!/bin/bash
set -e

echo "Starting HybridStack on Yeast..."
python scripts/reproduce_results.py --dataset yeast --pairing hadamard_abs --strategy same_compartment

echo "Starting External SOTAs on both Human and Yeast..."
python scripts/run_sota.py --methods all --dataset both --strategy same_compartment

echo "Starting Classical Baselines on both Human and Yeast..."
python scripts/run_baselines.py --dataset both --strategy same_compartment

echo "All complete!"
