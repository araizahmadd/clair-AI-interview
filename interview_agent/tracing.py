"""LangSmith tracing bootstrap for LangGraph."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def init_langsmith(*, load_env: bool = True) -> None:
    """
    Configure LangSmith when LANGCHAIN_TRACING_V2 is enabled.

    Set in .env or the shell:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=<your key>
      LANGCHAIN_PROJECT=interview-agent   # optional

    LangGraph runs are traced automatically when these env vars are present.
    """
    if load_env:
        load_dotenv()
    if os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    os.environ.setdefault("LANGCHAIN_PROJECT", "interview-agent")
    if not os.getenv("LANGCHAIN_API_KEY", "").strip():
        import warnings

        warnings.warn(
            "LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is missing; "
            "LangSmith traces will not upload.",
            stacklevel=2,
        )
