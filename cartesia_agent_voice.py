#!/usr/bin/env python3
"""
Shim CLI — Cartesia Calls API mic/speaker client lives under `interview_agent/voice/`.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from interview_agent.voice.cli import main

if __name__ == "__main__":
    main()
