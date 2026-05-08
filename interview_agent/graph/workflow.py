"""Compiled LangGraph: resume → company? → questions → optional Cartesia."""

from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, START, StateGraph

from interview_agent.graph.nodes import (
    node_analyze_interview_report,
    node_fetch_cartesia_transcript,
    node_generate_interview_questions,
    node_research_company,
    node_scan_resume_pdf,
    node_summarize_emotion_data,
    node_voice_interview,
)
from interview_agent.graph.state import InterviewPrepState


def route_after_resume(
    state: InterviewPrepState,
) -> Literal["company_research", "generate_questions"]:
    if (state.get("company_name") or "").strip():
        return "company_research"
    return "generate_questions"


def route_after_generate_questions(
    state: InterviewPrepState,
):
    if state.get("voice_interview_enabled"):
        return "voice_interview"
    return END


def route_after_voice_interview(
    state: InterviewPrepState,
):
    if state.get("voice_interview_completed"):
        return "fetch_cartesia_transcript"
    return END


def route_after_fetch_transcript(
    state: InterviewPrepState,
):
    if (state.get("interview_transcript_text") or "").strip():
        return "summarize_emotion_data"
    return END


def default_gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


def build_interview_graph():
    graph = StateGraph(InterviewPrepState)
    graph.add_node("scan_resume_pdf", node_scan_resume_pdf)
    graph.add_node("company_research", node_research_company)
    graph.add_node("generate_questions", node_generate_interview_questions)
    graph.add_node("voice_interview", node_voice_interview)
    graph.add_node("fetch_cartesia_transcript", node_fetch_cartesia_transcript)
    graph.add_node("summarize_emotion_data", node_summarize_emotion_data)
    graph.add_node("analyze_interview_report", node_analyze_interview_report)

    graph.add_edge(START, "scan_resume_pdf")
    graph.add_conditional_edges(
        "scan_resume_pdf",
        route_after_resume,
        {
            "company_research": "company_research",
            "generate_questions": "generate_questions",
        },
    )
    graph.add_edge("company_research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        route_after_generate_questions,
        {
            "voice_interview": "voice_interview",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "voice_interview",
        route_after_voice_interview,
        {
            "fetch_cartesia_transcript": "fetch_cartesia_transcript",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "fetch_cartesia_transcript",
        route_after_fetch_transcript,
        {
            "summarize_emotion_data": "summarize_emotion_data",
            END: END,
        },
    )
    graph.add_edge("summarize_emotion_data", "analyze_interview_report")
    graph.add_edge("analyze_interview_report", END)

    return graph.compile()
