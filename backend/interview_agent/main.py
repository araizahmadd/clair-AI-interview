"""CLI entry: resume + JD (+ company) → questions → optional voice interview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from interview_agent.graph.state import InterviewPrepState
from interview_agent.graph.workflow import build_interview_graph, default_gemini_model
from interview_agent.progress import log_node
from interview_agent.tracing import init_langsmith


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LangGraph pipeline: Gemini resume PDF + JD (+ Tavily company) "
            "→ interview questions (+ optional Cartesia voice mock interview). "
            "LangSmith: set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY."
        )
    )
    p.add_argument("--resume", required=True, help="Path to resume PDF")
    p.add_argument("--jd", default=None, help="Job description (inline)")
    p.add_argument(
        "--jd-file",
        default=None,
        help="Path to JD text/markdown",
    )
    p.add_argument(
        "--company",
        default=None,
        help="Optional company name (Tavily research)",
    )
    p.add_argument(
        "--model",
        default=default_gemini_model(),
        help=f"Gemini model (default: {default_gemini_model()})",
    )
    p.add_argument(
        "--voice",
        action="store_true",
        help=(
            "After generating questions, start Cartesia voice interview "
            "(requires PyAudio + CARTESIA_* env)."
        ),
    )
    p.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable webcam emotion monitoring during the Cartesia interview.",
    )
    p.add_argument(
        "--manual-stop",
        action="store_true",
        help="Disable silence auto-stop; use Enter or agent-ended websocket close only.",
    )
    p.add_argument(
        "--auto-stop-after-silence",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Auto-stop after this many seconds without mic or agent audio (default: env or 45).",
    )
    p.add_argument(
        "--cartesia-questions-metadata-only",
        action="store_true",
        help="Put questions only in metadata.session_questions for your Line agent.",
    )
    p.add_argument(
        "--cartesia-introduction",
        default=None,
        help="Optional session-only agent.introduction",
    )
    p.add_argument(
        "--cartesia-system-prompt-file",
        default=None,
        help="UTF-8 file merged into agent.system_prompt (questions appended unless --cartesia-questions-metadata-only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    init_langsmith(load_env=False)

    args = parse_args(argv)
    jd_parts: list[str] = []
    if args.jd:
        jd_parts.append(args.jd.strip())
    if args.jd_file:
        path = Path(args.jd_file).expanduser().resolve()
        jd_parts.append(path.read_text(encoding="utf-8").strip())
    job_description = "\n\n".join(jd_parts).strip()
    if not job_description:
        print("Provide job description via --jd and/or --jd-file.", file=sys.stderr)
        return 2

    initial: InterviewPrepState = {
        "pdf_path": str(Path(args.resume).expanduser().resolve()),
        "job_description": job_description,
        "company_name": args.company,
        "gemini_model": args.model,
        "voice_interview_enabled": bool(args.voice),
        "emotion_monitor_enabled": bool(args.voice and not args.no_emotion),
        "cartesia_manual_stop": bool(args.manual_stop),
        "cartesia_auto_stop_after_silence_seconds": args.auto_stop_after_silence,
        "cartesia_questions_metadata_only": bool(args.cartesia_questions_metadata_only),
        "cartesia_introduction": args.cartesia_introduction,
        "cartesia_system_prompt_file": args.cartesia_system_prompt_file,
        "errors": [],
    }

    log_node("main", "Starting interview LangGraph pipeline.")
    app = build_interview_graph()
    final = app.invoke(initial)
    log_node("main", "LangGraph pipeline finished.")

    errs = final.get("errors") or []
    if errs:
        print("Warnings / errors:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)

    questions = final.get("interview_questions") or []
    if not questions:
        print("No interview questions produced.", file=sys.stderr)
        return 1

    out: dict = {"interview_questions": questions}
    if args.voice:
        out["voice_interview_completed"] = bool(final.get("voice_interview_completed"))
        out["cartesia_session_id"] = final.get("cartesia_session_id")
        out["cartesia_session_started_at"] = final.get("cartesia_session_started_at")
        out["cartesia_call_id"] = final.get("cartesia_call_id")
        out["interview_transcript_path"] = final.get("interview_transcript_path")
        out["interview_event_log_path"] = final.get("interview_event_log_path")
        out["interview_mic_audio_path"] = final.get("interview_mic_audio_path")
        out["cartesia_official_call_path"] = final.get("cartesia_official_call_path")
        out["cartesia_stt_fallback_path"] = final.get("cartesia_stt_fallback_path")
        out["emotion_log_path"] = final.get("emotion_log_path")
        out["emotion_summary"] = final.get("emotion_summary")
        out["interview_report_path"] = final.get("interview_report_path")
        if final.get("interview_report_markdown"):
            out["interview_report_markdown"] = final.get("interview_report_markdown")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
