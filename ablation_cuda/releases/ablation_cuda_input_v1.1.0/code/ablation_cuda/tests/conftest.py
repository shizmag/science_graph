from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "back", REPO / "back/benchmarks/rag"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
