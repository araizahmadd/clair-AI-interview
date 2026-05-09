from __future__ import annotations

import threading
import time
from pathlib import Path
import sys
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

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
from interview_agent.graph.workflow import default_gemini_model

load_dotenv()

st.set_page_config(page_title="Clair Interview", page_icon="◼", layout="wide")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Space+Mono:wght@400;700&display=swap');
      .stApp {
        background: linear-gradient(180deg, #0d0d0d 0%, #151515 100%);
        color: #f2f2f2;
        font-family: 'Space Mono', monospace;
        font-size: 1.0625rem;
      }
      .hero {
        margin: 1rem auto 1.5rem auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        gap: 0.75rem;
        text-align: center;
      }
      .hero-title {
        font-family: 'Press Start 2P', monospace;
        font-size: clamp(3.5rem, 14vw, 8.25rem);
        letter-spacing: 0;
        line-height: 1.35;
        margin: 0 !important;
        text-transform: none;
        text-shadow:
          2px 2px 0 #1a1a1a,
          0 0 12px rgba(255, 255, 255, 0.12);
        -webkit-font-smoothing: none;
        image-rendering: pixelated;
      }
      .hero-subtitle {
        font-family: 'Press Start 2P', monospace;
        color: #b8b8b8;
        font-size: clamp(0.58rem, 2.5vw, 0.82rem);
        letter-spacing: 0.04em;
        line-height: 1.85;
        margin: 0 !important;
        text-transform: uppercase;
        max-width: 90vw;
        -webkit-font-smoothing: none;
      }
      [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        padding: 1rem !important;
      }
      [data-testid="stFileUploader"] button {
        min-height: 2.5rem !important;
        line-height: 1.2 !important;
      }
      .retro-title {
        font-family: 'Press Start 2P', monospace;
        font-size: clamp(1.1rem, 4.5vw, 2.15rem);
        text-transform: none;
        letter-spacing: 0;
        margin-bottom: 0.75rem;
        line-height: 1.65;
        -webkit-font-smoothing: none;
      }
      .retro-subtitle {
        font-family: 'Press Start 2P', monospace;
        color: #a8a8a8;
        font-size: clamp(0.48rem, 2vw, 0.62rem);
        letter-spacing: 0.06em;
        line-height: 1.85;
        margin-bottom: 1rem !important;
        -webkit-font-smoothing: none;
      }
      .pixel-heading {
        font-family: 'Press Start 2P', monospace;
        font-size: clamp(0.65rem, 2.6vw, 0.85rem);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0.75rem 0 0.5rem 0 !important;
        line-height: 1.85;
        color: #eaeaea;
        -webkit-font-smoothing: none;
      }
      div[data-testid="stButton"] button,
      .stApp button[kind="primary"],
      button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #e8e8e8 0%, #bdbdbd 100%) !important;
        color: #0d0d0d !important;
        border: 2px solid #f5f5f5 !important;
        border-radius: 6px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        box-shadow:
          0 0 0 1px #1a1a1a,
          0 4px 14px rgba(0, 0, 0, 0.45) !important;
        transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
      }
      div[data-testid="stButton"] button:hover,
      .stApp button[kind="primary"]:hover,
      button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(180deg, #ffffff 0%, #d0d0d0 100%) !important;
        border-color: #ffffff !important;
        box-shadow:
          0 0 0 1px #2a2a2a,
          0 0 18px rgba(255, 255, 255, 0.2),
          0 6px 18px rgba(0, 0, 0, 0.5) !important;
      }
      div[data-testid="stButton"] button:active,
      .stApp button[kind="primary"]:active,
      button[data-testid="baseButton-primary"]:active {
        background: #a8a8a8 !important;
        transform: translateY(1px);
      }
      div[data-testid="stButton"] button:disabled,
      .stApp button[kind="primary"]:disabled,
      button[data-testid="baseButton-primary"]:disabled {
        background: #3d3d3d !important;
        color: #8a8a8a !important;
        border-color: #525252 !important;
        box-shadow: none !important;
        opacity: 0.85 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "input"
    if "prep_state" not in st.session_state:
        st.session_state.prep_state = None
    if "runtime" not in st.session_state:
        st.session_state.runtime = {
            "thread": None,
            "steps": [],
            "done": False,
            "error": None,
            "final_state": None,
            "stop_requested": False,
            "voice_init_banner_until": 0.0,
        }


def _reset_flow() -> None:
    st.session_state.page = "input"
    st.session_state.prep_state = None
    for key in (
        "prep_running",
        "prep_pdf_bytes",
        "prep_pdf_name",
        "prep_jd",
        "prep_company",
    ):
        st.session_state.pop(key, None)
    st.session_state.runtime = {
        "thread": None,
        "steps": [],
        "done": False,
        "error": None,
        "final_state": None,
        "stop_requested": False,
        "voice_init_banner_until": 0.0,
    }


def _clear_prep_queue() -> None:
    st.session_state.pop("prep_running", None)
    for key in ("prep_pdf_bytes", "prep_pdf_name", "prep_jd", "prep_company"):
        st.session_state.pop(key, None)


def _merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if key == "errors":
            merged.setdefault("errors", [])
            merged["errors"] = [*merged["errors"], *(value or [])]
        else:
            merged[key] = value
    return merged


def _append_step(runtime: dict, step: str) -> None:
    runtime["steps"].append(step)


def _run_voice_and_reports(initial_state: InterviewPrepState, runtime: dict) -> None:
    state: dict = dict(initial_state)
    try:
        _append_step(runtime, "Starting live interview...")
        state = _merge(state, node_voice_interview(state))  # type: ignore[arg-type]
        if not state.get("voice_interview_completed"):
            raise RuntimeError("Voice interview did not complete successfully.")

        _append_step(runtime, "Processing results...")
        state = _merge(state, node_fetch_cartesia_transcript(state))  # type: ignore[arg-type]
        state = _merge(state, node_summarize_emotion_data(state))  # type: ignore[arg-type]

        _append_step(runtime, "Generating reports...")
        state = _merge(state, node_analyze_interview_report(state))  # type: ignore[arg-type]
        _append_step(runtime, "Interview complete.")
        runtime["final_state"] = state
    except Exception as exc:
        runtime["error"] = str(exc)
    finally:
        runtime["done"] = True


def _execute_prep_from_queue() -> None:
    st.caption("Preparing interview — buttons are locked.")
    rb = st.session_state.get("prep_pdf_bytes")
    rn = st.session_state.get("prep_pdf_name") or "resume.pdf"
    jd_txt = st.session_state.get("prep_jd") or ""

    if not isinstance(rb, (bytes, bytearray)) or not rb:
        _clear_prep_queue()
        st.error("Preparation data was missing. Please try Start Interview again.")
        return

    upload_dir = Path("backend/interview_agent/artifacts/ui_uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    resume_path = upload_dir / f"{uuid4()}_{rn}"
    resume_path.write_bytes(bytes(rb))
    stop_signal_file = upload_dir / f"stop_{uuid4()}.flag"
    if stop_signal_file.exists():
        stop_signal_file.unlink()

    state: InterviewPrepState = {
        "pdf_path": str(resume_path),
        "job_description": jd_txt,
        "company_name": st.session_state.get("prep_company"),
        "gemini_model": default_gemini_model(),
        "voice_interview_enabled": True,
        "emotion_monitor_enabled": True,
        "cartesia_manual_stop": True,
        "cartesia_stop_signal_file": str(stop_signal_file),
        "errors": [],
    }

    with st.status("Preparing interview...", expanded=True) as status:
        st.write("Scanning resume...")
        state = _merge(state, node_scan_resume_pdf(state))  # type: ignore[arg-type]

        if state.get("company_name"):
            st.write("Researching company context...")
            state = _merge(state, node_research_company(state))  # type: ignore[arg-type]

        st.write("Generating interview questions...")
        state = _merge(state, node_generate_interview_questions(state))  # type: ignore[arg-type]
        status.update(label="Preparation done", state="complete")

    _clear_prep_queue()

    if not state.get("interview_questions"):
        st.error("Could not generate interview questions.")
        if state.get("errors"):
            st.code("\n".join(state["errors"]))
        return

    st.session_state.prep_state = state
    st.session_state.runtime = {
        "thread": None,
        "steps": [],
        "done": False,
        "error": None,
        "final_state": None,
        "stop_requested": False,
        "voice_init_banner_until": 0.0,
    }
    st.session_state.page = "interview"
    st.rerun()


def _render_input_page() -> None:
    prep_busy = bool(st.session_state.get("prep_running"))

    if prep_busy:
        _execute_prep_from_queue()
        return

    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">CLAIR</p>
          <p class="hero-subtitle">YOUR INTERVIEW BUDDY</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="pixel-heading">Setup</p>', unsafe_allow_html=True)
    resume = st.file_uploader("Resume (PDF)", type=["pdf"])
    jd = st.text_area("Job Description", height=220, placeholder="Paste JD here...")
    company = st.text_input("Company Name (optional)")
    submitted = st.button(
        "Start Interview",
        use_container_width=True,
        type="primary",
    )

    if not submitted:
        return
    if resume is None:
        st.error("Please upload a resume PDF.")
        return
    if not jd.strip():
        st.error("Please provide a job description.")
        return

    st.session_state.prep_pdf_bytes = resume.getvalue()
    st.session_state.prep_pdf_name = resume.name
    st.session_state.prep_jd = jd.strip()
    company_s = company.strip()
    st.session_state.prep_company = company_s or None
    st.session_state.prep_running = True
    st.rerun()


def _render_interview_page() -> None:
    st.markdown('<p class="retro-title">Live Interview</p>', unsafe_allow_html=True)
    prep_state = st.session_state.prep_state
    runtime = st.session_state.runtime
    if not prep_state:
        st.session_state.page = "input"
        st.rerun()

    if runtime["thread"] is None:
        worker = threading.Thread(
            target=_run_voice_and_reports,
            args=(prep_state, runtime),
            daemon=True,
        )
        runtime["thread"] = worker
        # Show a short startup hint while Cartesia/emotion stack is booting.
        runtime["voice_init_banner_until"] = time.time() + 10
        worker.start()

    stop_sent = bool(runtime["stop_requested"])
    show_voice_init_banner = (
        not stop_sent
        and not runtime["done"]
        and not runtime["error"]
        and time.time() < float(runtime.get("voice_init_banner_until", 0.0))
    )
    if show_voice_init_banner:
        st.info(
            "Initializing voice interview... just a second. "
            "Camera is ready; audio session is starting in the background."
        )

    st.markdown('<p class="pixel-heading">Camera</p>', unsafe_allow_html=True)
    st.camera_input("Live camera", label_visibility="collapsed", disabled=stop_sent)
    if st.button(
        "End Interview",
        type="primary",
        use_container_width=True,
        disabled=stop_sent,
        key="end_interview_btn",
    ):
        runtime["stop_requested"] = True
        signal_path = Path(prep_state["cartesia_stop_signal_file"])
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.write_text("stop", encoding="utf-8")
        _append_step(runtime, "End interview requested by user.")
        st.session_state.page = "processing"
        st.rerun()

    st.caption("When finished, click End Interview to close camera and continue.")


def _render_processing_page() -> None:
    runtime = st.session_state.runtime
    st.markdown('<div class="retro-title">Processing Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="retro-subtitle">Post interview analysis and report</div>', unsafe_allow_html=True)

    st.markdown('<p class="pixel-heading">Next Steps</p>', unsafe_allow_html=True)
    if runtime["steps"]:
        for idx, step in enumerate(runtime["steps"]):
            if not runtime["done"] and idx == len(runtime["steps"]) - 1:
                st.write(f"⏳ {step}")
            else:
                st.write(f"✅ {step}")
    else:
        st.write("⏳ Waiting for processing to begin...")

    if runtime["error"]:
        st.error(runtime["error"])
        if st.button("Start New Interview", key="restart_after_error"):
            _reset_flow()
            st.rerun()
        return

    if runtime["done"]:
        final_state = runtime["final_state"] or {}
        report = final_state.get("interview_report_markdown")
        report_path = final_state.get("interview_report_path")
        if report:
            st.markdown('<div class="retro-title">Interview Report</div>', unsafe_allow_html=True)
            st.markdown(report)
            if report_path:
                st.caption(f"Saved at: {report_path}")
        else:
            st.warning("Interview ended, but no report was generated.")
            if final_state.get("errors"):
                st.code("\n".join(final_state["errors"]))
        if st.button("Start New Interview", key="restart_after_done"):
            _reset_flow()
            st.rerun()
        return

    st.caption("Processing... please keep this tab open.")
    time.sleep(1)
    st.rerun()


def main() -> None:
    _init_state()
    if st.session_state.page == "input":
        _render_input_page()
    elif st.session_state.page == "interview":
        _render_interview_page()
    else:
        _render_processing_page()


if __name__ == "__main__":
    main()
