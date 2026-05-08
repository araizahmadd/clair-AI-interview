"""LangGraph shared state."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired

from typing_extensions import TypedDict


class InterviewPrepState(TypedDict):
    pdf_path: str
    job_description: str
    company_name: NotRequired[str | None]
    gemini_model: str
    voice_interview_enabled: NotRequired[bool]
    emotion_monitor_enabled: NotRequired[bool]
    cartesia_manual_stop: NotRequired[bool]
    cartesia_auto_stop_after_silence_seconds: NotRequired[float]
    cartesia_questions_metadata_only: NotRequired[bool]
    cartesia_system_prompt_file: NotRequired[str | None]
    cartesia_introduction: NotRequired[str | None]
    cartesia_stop_signal_file: NotRequired[str | None]
    cartesia_session_id: NotRequired[str | None]
    cartesia_session_started_at: NotRequired[str | None]
    cartesia_agent_id: NotRequired[str | None]
    cartesia_call_id: NotRequired[str | None]
    resume_summary: NotRequired[str]
    company_context: NotRequired[str | None]
    interview_questions: NotRequired[list[str]]
    uploaded_file_resource_name: NotRequired[str | None]
    voice_interview_completed: NotRequired[bool]
    interview_transcript_path: NotRequired[str | None]
    interview_event_log_path: NotRequired[str | None]
    interview_mic_audio_path: NotRequired[str | None]
    cartesia_official_call_path: NotRequired[str | None]
    cartesia_stt_fallback_path: NotRequired[str | None]
    interview_transcript_text: NotRequired[str | None]
    emotion_log_path: NotRequired[str | None]
    emotion_summary: NotRequired[str | None]
    emotion_sample_count: NotRequired[int]
    interview_report_markdown: NotRequired[str | None]
    interview_report_path: NotRequired[str | None]
    errors: Annotated[list[str], operator.add]
