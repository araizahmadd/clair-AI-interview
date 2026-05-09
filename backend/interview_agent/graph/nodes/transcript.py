"""Fetch the official Cartesia call transcript after the voice session ends."""

from __future__ import annotations

from interview_agent.graph.state import InterviewPrepState
from interview_agent.progress import log_node
from interview_agent.voice.cartesia import (
    CartesiaConfigError,
    fetch_official_call_transcript,
    transcribe_mic_audio_fallback,
)


def node_fetch_cartesia_transcript(state: InterviewPrepState) -> dict:
    log_node("fetch_cartesia_transcript", "Fetching official Cartesia call transcript.")
    session_id = (state.get("cartesia_session_id") or "").strip()
    agent_id = (state.get("cartesia_agent_id") or "").strip()
    session_started_at = state.get("cartesia_session_started_at")
    mic_audio_path = state.get("interview_mic_audio_path")
    if not session_id or not agent_id:
        log_node("fetch_cartesia_transcript", "Skipped: missing session id or agent id.")
        if mic_audio_path:
            return _fallback_from_mic_audio(mic_audio_path, session_id or "unknown")
        return {"errors": ["Transcript skipped: missing Cartesia ids and no mic audio fallback."]}

    try:
        official = fetch_official_call_transcript(
            agent_id=agent_id,
            session_id=session_id,
            session_started_at=session_started_at,
        )
        log_node(
            "fetch_cartesia_transcript",
            f"Fetched official transcript for call {official.get('call_id')}.",
        )
        return {
            "cartesia_call_id": official.get("call_id"),
            "interview_transcript_path": official.get("transcript_path"),
            "cartesia_official_call_path": official.get("raw_call_path"),
            "interview_transcript_text": official.get("transcript_text") or "",
        }
    except CartesiaConfigError as exc:
        # Keep any best-effort websocket transcript already on state, but surface why
        # the official Cartesia transcript could not be fetched.
        log_node("fetch_cartesia_transcript", f"Failed: {exc}")
        fallback = _fallback_from_mic_audio(mic_audio_path, session_id)
        fallback.setdefault("errors", [])
        fallback["errors"] = [
            f"Official Cartesia transcript fetch failed: {exc}",
            *fallback["errors"],
        ]
        return fallback
    except Exception as exc:
        log_node("fetch_cartesia_transcript", f"Failed: {exc}")
        fallback = _fallback_from_mic_audio(mic_audio_path, session_id)
        fallback.setdefault("errors", [])
        fallback["errors"] = [
            f"Official Cartesia transcript fetch failed: {exc}",
            *fallback["errors"],
        ]
        return fallback


def _fallback_from_mic_audio(audio_path: str | None, session_id: str) -> dict:
    if not audio_path:
        return {"errors": ["STT fallback skipped: no recorded mic audio path."]}
    try:
        log_node("fetch_cartesia_transcript", "Using recorded mic audio STT fallback.")
        fallback = transcribe_mic_audio_fallback(
            audio_path=audio_path,
            session_id=session_id,
        )
        return {
            "interview_transcript_path": fallback.get("transcript_path"),
            "cartesia_stt_fallback_path": fallback.get("raw_stt_path"),
            "interview_transcript_text": fallback.get("transcript_text") or "",
        }
    except Exception as exc:
        log_node("fetch_cartesia_transcript", f"STT fallback failed: {exc}")
        return {"errors": [f"STT fallback transcript failed: {exc}"]}
