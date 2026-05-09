"""Summarize emotion-monitor CSV before report generation."""

from __future__ import annotations

from interview_agent.emotion.scanner import summarize_emotion_csv
from interview_agent.graph.state import InterviewPrepState
from interview_agent.progress import log_node


def node_summarize_emotion_data(state: InterviewPrepState) -> dict:
    log_node("summarize_emotion_data", "Summarizing emotion monitor CSV.")
    path = state.get("emotion_log_path")
    if not path:
        log_node("summarize_emotion_data", "No emotion CSV path found.")
        return {"emotion_summary": "No emotion monitor data was collected."}

    result = summarize_emotion_csv(path)
    log_node(
        "summarize_emotion_data",
        f"Emotion summary ready from {result.sample_count} samples.",
    )
    out = {
        "emotion_log_path": result.csv_path,
        "emotion_summary": result.summary,
        "emotion_sample_count": result.sample_count,
    }
    if result.error:
        out["errors"] = [f"Emotion monitor summary: {result.error}"]
    return out
