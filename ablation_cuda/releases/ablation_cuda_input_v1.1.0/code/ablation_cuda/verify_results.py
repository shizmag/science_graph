#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
if CODE.is_dir():
    sys.path.insert(0, str(CODE))
else:
    sys.path.insert(0, str(ROOT.parent))

from ablation_cuda.verify import main

if __name__ == "__main__":
    main()
