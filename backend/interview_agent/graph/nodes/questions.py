"""Generate five interview questions from resume + JD (+ company)."""

from __future__ import annotations

from google import genai
from google.genai import types

from interview_agent.gemini import generate_content_with_fallback
from interview_agent.graph.state import InterviewPrepState
from interview_agent.keys import gemini_api_key
from interview_agent.progress import log_node
from interview_agent.schemas import InterviewQuestions


def node_generate_interview_questions(state: InterviewPrepState) -> dict:
    log_node("generate_questions", "Generating five interview questions with Gemini.")
    client = genai.Client(api_key=gemini_api_key())
    model = state["gemini_model"]
    jd = state["job_description"].strip()
    resume = (state.get("resume_summary") or "").strip()
    company_blob = (state.get("company_context") or "").strip()
    company_name = (state.get("company_name") or "").strip()

    context_parts = [
        "## Resume summary (from PDF)",
        resume or "(Resume scan unavailable — infer cautiously from JD only.)",
        "",
        "## Job description",
        jd or "(No job description provided.)",
    ]
    if company_blob:
        context_parts.extend(
            [
                "",
                f"## Company context{f' ({company_name})' if company_name else ''}",
                company_blob,
            ]
        )

    prompt = "\n".join(context_parts)
    instructions = (
        "You are an experienced hiring manager. Using ONLY the materials above, propose "
        "exactly 3 interview questions for THIS candidate and THIS role. Keep them short and easy to answer.\n"
        "Requirements:\n"
        "- Questions must connect explicit resume evidence to JD responsibilities.\n"
        "- Prefer behavioral + technical depth probes; avoid generic clichés.\n"
        "- If company context is included, weave in at most two questions that reflect "
        "realistic expectations for that employer (no rumors).\n"
        "- If resume summary is missing or thin, ask questions that still validate core JD skills.\n"
        "Return JSON matching the schema."
    )

    try:
        response, _, warnings = generate_content_with_fallback(
            client,
            primary_model=model,
            contents=[prompt, instructions],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=InterviewQuestions.model_json_schema(),
            ),
        )
        if not response.text:
            log_node("generate_questions", "Gemini returned empty question output.")
            return {"errors": ["Gemini returned empty text for interview questions."]}
        parsed = InterviewQuestions.model_validate_json(response.text)
        log_node("generate_questions", f"Generated {len(parsed.questions)} questions.")
        return {"interview_questions": list(parsed.questions), "errors": warnings}
    except Exception as exc:
        log_node("generate_questions", f"Failed: {exc}")
        return {"errors": [f"Interview question generation failed: {exc}"]}
