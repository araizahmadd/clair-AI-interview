#!/usr/bin/env python3
"""Standalone CLI for interview emotion monitoring."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from interview_agent.emotion.scanner import run_emotion_scanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run webcam emotion detection.")
    parser.add_argument("--output", default="emotion_log.csv", help="CSV output path.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    stop = threading.Event()
    try:
        result = run_emotion_scanner(
            csv_path=args.output,
            stop_event=stop,
            camera_index=args.camera_index,
            show_window=not args.no_window,
        )
        print(result.summary)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")


if __name__ == "__main__":
    main()
