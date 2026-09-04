"""
Shared pytest configuration.

Puts the project root on `sys.path` so `src` and `app` import the same way they
do when the pipelines run as modules, regardless of where pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Any test that touches the report or EDA path must not try to open a window.
matplotlib.use("Agg")
