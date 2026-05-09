"""Analyze interview transcript question-by-question and write markdown report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from google import genai

from interview_agent.gemini import generate_content_with_fallback
from interview_agent.graph.state import InterviewPrepState
from interview_agent.keys import gemini_api_key
from interview_agent.progress import log_node


def node_analyze_interview_report(state: InterviewPrepState) -> dict:
    log_node("analyze_interview_report", "Generating markdown interview report with Gemini.")
    questions = list(state.get("interview_questions") or [])
    transcript = (state.get("interview_transcript_text") or "").strip()
    emotion_summary = (state.get("emotion_summary") or "").strip()
    if not questions:
        log_node("analyze_interview_report", "Skipped: no questions found.")
        return {"errors": ["Interview report skipped: no interview questions found."]}
    if not transcript:
        log_node("analyze_interview_report", "Skipped: no transcript found.")
        return {"errors": ["Interview report skipped: no transcript captured from voice interview."]}

    model = state["gemini_model"]
    client = genai.Client(api_key=gemini_api_key())

    question_block = "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions, start=1))
    prompt = (
        "You are an interview coach creating candidate-facing feedback after a mock interview.\n"
        "Write directly to the candidate using 'you'. Be honest, specific, practical, and encouraging.\n"
        "Use the exact interview questions and transcript script below.\n\n"
        "Return Markdown with this structure:\n"
        "# Interview Feedback Report\n"
        "## Overall Feedback\n"
        "- 4-6 bullets on how the candidate performed, what came through well, and what needs work.\n"
        "## Scorecard\n"
        "Include a Markdown table with these rows and scores out of 10: Technical Depth, Relevance to Role, Communication Clarity, Specificity of Examples, Confidence & Presence, Overall Interview Score.\n"
        "For each row include: Score, Reason, and One Improvement.\n"
        "## Key Strengths To Keep\n"
        "- 3-5 bullets on behaviors, examples, skills, or communication patterns worth continuing.\n"
        "## Biggest Improvement Areas\n"
        "- 3-5 bullets explaining the highest-impact changes to make before the real interview.\n"
        "## Question-by-Question Feedback\n"
        "For each question 1..N include:\n"
        "### QN: <short title>\n"
        "- Original question: ...\n"
        "- What you answered: ...\n"
        "- What worked: ...\n"
        "- What to improve: ...\n"
        "- Better answer strategy: ...\n"
        "- Score (1-10): ...\n"
        "- Why this score: ...\n"
        "## Tips To Improve\n"
        "- 5-8 concrete practice tips, phrased as actions.\n"
        "## Things To Avoid\n"
        "- 5-8 pitfalls to avoid, based on the transcript or likely risks.\n"
        "## Suggested Practice Plan\n"
        "- A short 3-day practice plan with daily actions.\n"
        "## Final Readiness Assessment\n"
        "- Readiness level: ...\n"
        "- Overall score (1-10): ...\n"
        "- One-sentence summary: ...\n"
        "- Top 3 next steps: ...\n"
        "## Nonverbal / Emotion Signal\n"
        "Write this section with extra care. Automated facial-emotion scores are noisy and often wrong; "
        "they are not a measure of the candidate's true feelings, personality, or fitness for the role.\n"
        "- Open with 2-4 sentences of general caution: lighting, camera angle, glasses, compression, "
        "resting face, and concentration can be misread; low average model confidence "
        "(for example below ~0.6) means labels are especially unreliable.\n"
        "- Parse the emotion monitor summary. If it reports average confidence, comment on whether it is "
        "relatively low or high and what that implies for how strongly to trust the distribution.\n"
        "- If no emotion data or no usable samples: say so briefly and skip invented interpretations.\n"
        "- Otherwise, for **each distinct emotion label** that appears meaningfully in the distribution "
        "(not \"no_face\" / \"face_detected\"), add a short bullet that covers:\n"
        "  - What the monitor suggests (percentage/confidence context).\n"
        "  - **Likely alternative reads** on camera—map labels to plausible benign causes, for example:\n"
        "    - Sadness → resting face, fatigue, focused listening, mild stress, not necessarily low mood.\n"
        "    - Fear / anxiety-like cues → anticipation, stakes, intense concentration, processing a hard question.\n"
        "    - Anger / disgust-like cues → determination, frustration with tech or retakes, strong emphasis, "
        "skepticism while thinking, asymmetry from speech or shadows—not hostility.\n"
        "    - Happiness / joy → polite or nervous smiling, relief after answering; tie to \"confidence\" only "
        "if the transcript and delivery sound confident—do not equate a smile metric with self-assurance alone.\n"
        "    - Surprise → eyebrow raise while thinking, reacting to wording, emphasis—not necessarily shock.\n"
        "    - Neutral → calm professionalism, seriousness, or flat affect under poor capture.\n"
        "  - One line reconciling with **spoken delivery**: if the transcript reads composed or confident, "
        "state that voice and content are stronger signals than the face label and should weigh more.\n"
        "- Never diagnose mental health. Never claim the candidate \"was\" a given emotion; use wording like "
        "\"the model tended toward,\" \"may reflect,\" or \"can be confused with.\"\n"
        "- Close by recommending the candidate prioritize actionable feedback from answers above over facial labels.\n\n"
        "Ground feedback in transcript evidence. Do not invent answers the candidate did not give. "
        "If the transcript is sparse or empty, explain that the main improvement is to answer audibly and fully, "
        "then provide general preparation tips for the asked questions. Use emotion data cautiously.\n\n"
        f"Questions:\n{question_block}\n\n"
        f"Transcript:\n{transcript}\n"
        f"\nEmotion monitor summary:\n{emotion_summary or '(No emotion data available.)'}\n"
    )

    try:
        response, _, warnings = generate_content_with_fallback(
            client,
            primary_model=model,
            contents=prompt,
        )
        report_md = (response.text or "").strip()
        if not report_md:
            return {"errors": ["Interview report generation failed: empty model response."]}

        base_dir = Path("backend/interview_agent/artifacts/reports").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = base_dir / f"interview_report_{stamp}.md"
        report_path.write_text(report_md, encoding="utf-8")
        log_node("analyze_interview_report", f"Report written to {report_path}.")

        return {
            "interview_report_markdown": report_md,
            "interview_report_path": str(report_path),
            "errors": warnings,
        }
    except Exception as exc:
        log_node("analyze_interview_report", f"Failed: {exc}")
        return {"errors": [f"Interview report generation failed: {exc}"]}
