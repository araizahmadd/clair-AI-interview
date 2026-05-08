#!/usr/bin/env python3
"""
Shim CLI — implementation lives under `interview_agent/`.
Run from repo root: python interview_prep_langgraph.py ...
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from interview_agent.main import main

if __name__ == "__main__":
    raise SystemExit(main())
