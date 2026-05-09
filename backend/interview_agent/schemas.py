"""Structured outputs for Gemini."""

from pydantic import BaseModel, Field


class ResumeScanResult(BaseModel):
    candidate_summary: str = Field(
        description="2–4 sentences: who they are and what they optimize for."
    )
    top_skills: list[str] = Field(
        description="Hard and soft skills most relevant to hiring.", max_length=24
    )
    experience_highlights: list[str] = Field(
        description="Short bullets of roles, scope, and measurable outcomes.",
        max_length=12,
    )
    education_and_credentials: list[str] = Field(
        default_factory=list,
        description="Degrees, certifications, notable coursework.",
        max_length=8,
    )


class InterviewQuestions(BaseModel):
    questions: list[str] = Field(
        ...,
        description="Five distinct, specific interview questions.",
        min_length=5,
        max_length=5,
    )
