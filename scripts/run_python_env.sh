#!/bin/bash
# Wrapper to run python within the project's conda environment
source "/home/hitokiri/miniconda3/etc/profile.d/conda.sh"
conda activate ppis_env
python "$@"
