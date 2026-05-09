"""Gemini model fallback helpers."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_FALLBACK_MODELS = ("gemini-2.5-pro", "gemini-2.5-flash")


def gemini_model_candidates(primary_model: str) -> list[str]:
    fallback_raw = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
    fallbacks = (
        [m.strip() for m in fallback_raw.split(",") if m.strip()]
        if fallback_raw
        else list(DEFAULT_FALLBACK_MODELS)
    )
    out: list[str] = []
    for model in [primary_model, *fallbacks]:
        if model and model not in out:
            out.append(model)
    return out


def is_transient_gemini_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "503",
            "unavailable",
            "high demand",
            "resource_exhausted",
            "rate limit",
            "temporarily",
        )
    )


def generate_content_with_fallback(
    client: Any,
    *,
    primary_model: str,
    contents: Any,
    config: Any | None = None,
) -> tuple[Any, str, list[str]]:
    warnings: list[str] = []
    last_exc: Exception | None = None

    for model in gemini_model_candidates(primary_model):
        try:
            kwargs: dict[str, Any] = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            if model != primary_model:
                warnings.append(
                    f"Gemini primary model {primary_model!r} was unavailable; used fallback {model!r}."
                )
            return response, model, warnings
        except Exception as exc:
            last_exc = exc
            if model == primary_model and is_transient_gemini_error(exc):
                warnings.append(
                    f"Gemini primary model {primary_model!r} failed transiently: {exc}"
                )
                continue
            if model != primary_model and is_transient_gemini_error(exc):
                warnings.append(f"Gemini fallback model {model!r} failed transiently: {exc}")
                continue
            raise

    assert last_exc is not None
    raise last_exc
