"""Cartesia voice interview node (consumes Gemini-generated questions)."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from interview_agent.emotion.scanner import BackgroundEmotionMonitor
from interview_agent.graph.state import InterviewPrepState
from interview_agent.progress import log_node
from interview_agent.voice.cartesia import CartesiaConfigError, run_voice_interview_sync


def node_voice_interview(state: InterviewPrepState) -> dict:
    log_node("voice_interview", "Preparing Cartesia voice interview.")
    questions = list(state.get("interview_questions") or [])
    if not questions:
        log_node("voice_interview", "Skipped because no interview questions are available.")
        return {
            "errors": ["Voice interview skipped: no interview_questions present."],
            "voice_interview_completed": False,
        }

    q_meta_only = bool(state.get("cartesia_questions_metadata_only"))
    manual_stop = bool(state.get("cartesia_manual_stop"))
    auto_stop_after_silence = state.get("cartesia_auto_stop_after_silence_seconds")
    introduction = state.get("cartesia_introduction")
    prompt_file_raw = state.get("cartesia_system_prompt_file")
    stop_signal_file_raw = state.get("cartesia_stop_signal_file")
    prompt_file = (
        Path(prompt_file_raw).expanduser().resolve()
        if prompt_file_raw
        else None
    )
    stop_signal_file = (
        Path(stop_signal_file_raw).expanduser().resolve()
        if stop_signal_file_raw
        else None
    )

    intro_s = introduction.strip() if isinstance(introduction, str) else None
    session_id = f"interview-{uuid4()}"
    emotion_enabled = bool(state.get("emotion_monitor_enabled", True))
    emotion_monitor: BackgroundEmotionMonitor | None = None
    emotion_result = None

    if emotion_enabled:
        log_node("voice_interview", "Starting background emotion monitor.")
        emotion_csv = (
            Path("interview_agent/artifacts/emotion") / session_id / "emotion_log.csv"
        )
        show_window = os.getenv("EMOTION_SHOW_WINDOW", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        camera_index = int(os.getenv("EMOTION_CAMERA_INDEX", "0"))
        emotion_monitor = BackgroundEmotionMonitor(
            csv_path=emotion_csv,
            camera_index=camera_index,
            show_window=show_window,
        )
        emotion_monitor.start()
    else:
        log_node("voice_interview", "Emotion monitor disabled.")

    try:
        log_node(
            "voice_interview",
            "Starting live Cartesia session. Press Enter in the terminal when the interview is finished.",
        )
        voice_result = run_voice_interview_sync(
            questions,
            introduction=intro_s,
            system_prompt_file=prompt_file if prompt_file and prompt_file.is_file() else None,
            questions_metadata_only=q_meta_only,
            session_id=session_id,
            manual_stop=manual_stop,
            auto_stop_after_silence_seconds=auto_stop_after_silence,
            stop_signal_file=stop_signal_file,
        )
        transcript_lines = list(voice_result.get("transcript_lines") or [])
        if emotion_monitor:
            log_node("voice_interview", "Stopping emotion monitor.")
            emotion_result = emotion_monitor.stop()

        log_node("voice_interview", "Cartesia voice session ended.")
        result = {
            "voice_interview_completed": True,
            "cartesia_session_id": voice_result.get("session_id") or session_id,
            "cartesia_session_started_at": voice_result.get("session_started_at"),
            "cartesia_agent_id": voice_result.get("agent_id"),
            "interview_transcript_path": voice_result.get("transcript_path"),
            "interview_event_log_path": voice_result.get("event_log_path"),
            "interview_mic_audio_path": voice_result.get("mic_audio_path"),
            "interview_transcript_text": "\n".join(transcript_lines).strip(),
        }
        if emotion_result:
            result.update(
                {
                    "emotion_log_path": emotion_result.csv_path,
                    "emotion_summary": emotion_result.summary,
                    "emotion_sample_count": emotion_result.sample_count,
                }
            )
            if emotion_result.error:
                result["errors"] = [f"Emotion monitor: {emotion_result.error}"]
        return result
    except CartesiaConfigError as exc:
        if emotion_monitor:
            log_node("voice_interview", "Stopping emotion monitor after Cartesia error.")
            emotion_result = emotion_monitor.stop()
        log_node("voice_interview", f"Failed: {exc}")
        return {
            "errors": [f"Cartesia voice interview: {exc}"],
            "voice_interview_completed": False,
            **(
                {
                    "emotion_log_path": emotion_result.csv_path,
                    "emotion_summary": emotion_result.summary,
                    "emotion_sample_count": emotion_result.sample_count,
                }
                if emotion_result
                else {}
            ),
        }
    except Exception as exc:
        if emotion_monitor:
            log_node("voice_interview", "Stopping emotion monitor after voice error.")
            emotion_result = emotion_monitor.stop()
        log_node("voice_interview", f"Failed: {exc}")
        return {
            "errors": [f"Voice interview failed: {exc}"],
            "voice_interview_completed": False,
            **(
                {
                    "emotion_log_path": emotion_result.csv_path,
                    "emotion_summary": emotion_result.summary,
                    "emotion_sample_count": emotion_result.sample_count,
                }
                if emotion_result
                else {}
            ),
        }
