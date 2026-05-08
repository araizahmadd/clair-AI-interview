#!/usr/bin/env python3
"""Standalone shim for the interview emotion monitor."""

from __future__ import annotations

import argparse
import threading

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
