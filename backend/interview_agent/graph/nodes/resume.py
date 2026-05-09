"""Resume PDF extraction via Gemini File API."""

from __future__ import annotations

from pathlib import Path

from google import genai
from google.genai import types

from interview_agent.gemini import generate_content_with_fallback
from interview_agent.graph.state import InterviewPrepState
from interview_agent.keys import gemini_api_key
from interview_agent.progress import log_node
from interview_agent.schemas import ResumeScanResult


def _format_resume_summary(scan: ResumeScanResult) -> str:
    lines = [
        scan.candidate_summary.strip(),
        "",
        "Skills:",
        *[f"- {s}" for s in scan.top_skills],
        "",
        "Experience:",
        *[f"- {e}" for e in scan.experience_highlights],
        "",
        "Education / credentials:",
        *[f"- {x}" for x in scan.education_and_credentials],
    ]
    return "\n".join(lines).strip()


def node_scan_resume_pdf(state: InterviewPrepState) -> dict:
    log_node("scan_resume_pdf", "Starting Gemini resume PDF scan.")
    pdf = Path(state["pdf_path"]).expanduser().resolve()
    if not pdf.is_file():
        log_node("scan_resume_pdf", f"Resume PDF not found: {pdf}")
        return {
            "errors": [f"Resume PDF not found: {pdf}"],
            "resume_summary": "",
        }

    client = genai.Client(api_key=gemini_api_key())
    model = state["gemini_model"]
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(pdf))
        prompt = (
            "You are extracting structured facts from this resume PDF for interview prep. "
            "Be faithful to the document; do not invent employers, dates, or skills."
        )
        response, _, warnings = generate_content_with_fallback(
            client,
            primary_model=model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ResumeScanResult.model_json_schema(),
            ),
        )
        if not response.text:
            log_node("scan_resume_pdf", "Gemini returned an empty resume scan.")
            return {
                "errors": ["Gemini returned empty text for resume scan."],
                "resume_summary": "",
                "uploaded_file_resource_name": getattr(uploaded, "name", None),
            }
        scan = ResumeScanResult.model_validate_json(response.text)
        log_node("scan_resume_pdf", "Completed resume scan.")
        return {
            "resume_summary": _format_resume_summary(scan),
            "uploaded_file_resource_name": uploaded.name,
            "errors": warnings,
        }
    except Exception as exc:
        log_node("scan_resume_pdf", f"Failed: {exc}")
        return {
            "errors": [f"Resume scan failed: {exc}"],
            "resume_summary": "",
            "uploaded_file_resource_name": uploaded.name if uploaded else None,
        }
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
