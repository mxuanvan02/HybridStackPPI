#!/usr/bin/env python3
"""Compatibility wrapper for reorganized baseline modules."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.sota_esm2_mlp import *  # noqa: F401,F403
