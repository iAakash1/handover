"""Make the `_dkb` helper importable regardless of the runner's CWD."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
