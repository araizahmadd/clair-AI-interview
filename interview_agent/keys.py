"""API key helpers."""

from __future__ import annotations

import os


def gemini_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    return key


def tavily_api_key() -> str:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is required when a company name is provided."
        )
    return key
