"""Small console progress logger for CLI runs."""

from __future__ import annotations

from datetime import datetime


def log_node(node: str, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] [{node}] {message}", flush=True)
