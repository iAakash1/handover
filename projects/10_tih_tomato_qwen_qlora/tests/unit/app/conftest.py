"""Make the repo-root ``app`` package importable for the demo-app tests.

``app/`` lives at the repo root (not under ``src/``), so it isn't on the path
the way the installed ``plantdx`` package is. These tests are for the Streamlit
demo layer only; they never touch training/evaluation behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
