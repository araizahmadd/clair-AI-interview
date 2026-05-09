#!/usr/bin/env python3
"""
CLI shim for interview prep pipeline.
Run from repo root: python scripts/interview_prep.py ...
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from interview_agent.main import main

if __name__ == "__main__":
    raise SystemExit(main())
