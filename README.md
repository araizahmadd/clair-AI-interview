# Clair 2 - AI Interview Copilot

Clair 2 is a Python application for mock interview preparation and live interview simulation.  
It combines resume parsing, job description analysis, optional company research, AI question generation, voice interview execution, emotion tracking, and report generation.

## What This Repository Includes

- `backend/interview_agent/`: Main LangGraph-based interview workflow.
- `frontend/frontend_streamlit.py`: Streamlit UI for end-to-end interview sessions.
- `scripts/`: CLI entrypoints (interview prep, Cartesia voice, ElevenLabs chat, emotion scanner).
- `backend/interview_agent/artifacts/`: Generated transcripts, reports, session logs, and emotion outputs.

## Features

- Resume PDF analysis with Gemini.
- Job description alignment and question generation.
- Optional company context research using Tavily.
- Optional live voice mock interview using Cartesia.
- Optional emotion monitoring during interview sessions.
- Post-interview transcript, emotion summary, and markdown report generation.
- Streamlit UI for interactive usage.

## Requirements

- Python 3.10+ (3.11/3.12 recommended).
- macOS/Linux/Windows.
- API keys depending on features used:
  - Gemini (`GOOGLE_API_KEY`) for resume and question generation.
  - Tavily (`TAVILY_API_KEY`) for company research.
  - Cartesia (`CARTESIA_API_KEY`, `CARTESIA_AGENT_ID`) for live voice interviews.
  - ElevenLabs (`ELEVENLABS_API_KEY`, `ELEVENLABS_AGENT_ID`) for ElevenLabs scripts.
  - LangSmith (`LANGCHAIN_API_KEY`) if you want tracing.

For microphone/speaker support, install PortAudio before PyAudio.

## Quick Start

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Configure environment:

```bash
cp .env.example .env
```

Fill `.env` with the keys needed for your selected workflow.

## Run The Streamlit App

```bash
streamlit run frontend/frontend_streamlit.py
```

This launches the full interview flow:
- upload resume PDF
- paste job description
- optionally provide company name
- start interview, then review generated report

## Workflow Diagram

```mermaid
flowchart TD
    A[User Starts App] --> B{Interface}
    B -->|Web UI| C[Streamlit: frontend/frontend_streamlit.py]
    B -->|CLI| D[scripts/interview_prep.py]

    C --> E[Upload Resume PDF + Paste JD + Optional Company]
    D --> E

    E --> F[LangGraph Workflow: backend/interview_agent/graph/workflow.py]

    F --> G[Node: Scan Resume PDF]
    G --> H{Company Provided?}
    H -->|Yes| I[Node: Tavily Company Research]
    H -->|No| J[Skip Research]
    I --> K[Node: Generate Interview Questions]
    J --> K

    K --> L{Voice Enabled?}
    L -->|No| M[Return Interview Questions JSON]
    L -->|Yes| N[Node: Voice Interview (Cartesia)]

    N --> O[Start Cartesia WebSocket Session]
    N --> P[Start Background Emotion Monitor]
    O --> Q[Capture Mic + Play Agent Audio]
    P --> R[Write Emotion CSV Logs]

    Q --> S{Interview Completed?}
    S -->|Yes| T[Fetch Official Cartesia Transcript]
    S -->|No/Error| U[Save Errors + Partial Artifacts]

    T --> V[Summarize Emotion Data]
    V --> W[Generate Final Interview Report]
    W --> X[Save Artifacts]

    X --> Y[artifacts/sessions/: events, transcript, audio]
    X --> Z[artifacts/emotion/: emotion_log.csv]
    X --> AA[artifacts/reports/: markdown report]

    AA --> AB[Display in Streamlit or CLI Output]
```

## Run The CLI Interview Pipeline

```bash
python -m interview_agent.main \
  --resume "/path/to/resume.pdf" \
  --jd-file "/path/to/jd.txt"
```

Optional flags:
- `--company "Company Name"` to include company research.
- `--voice` to run Cartesia voice interview after question generation.
- `--no-emotion` to disable emotion monitoring when using `--voice`.

## Other Entrypoints

- `python scripts/interview_prep.py`  
  Alternate prep flow script.
- `python scripts/cartesia_agent_voice.py`  
  Direct Cartesia voice test script.
- `python scripts/elevenlabs_agent_chat.py`  
  ElevenLabs conversational script.
- `python scripts/emotion_scanner.py`  
  Standalone emotion monitor test script.

## Project Structure (High Level)

```text
.
├── backend/
│   └── interview_agent/
│       ├── graph/            # LangGraph nodes, state, workflow
│       ├── voice/            # Cartesia voice integration
│       ├── emotion/          # Emotion scanner integration
│       └── artifacts/        # Runtime outputs (reports, logs, transcripts)
├── frontend/
│   └── frontend_streamlit.py # Web UI
├── scripts/                  # All CLI entrypoints
└── requirements.txt
```

## Security And Git Hygiene

- Do not commit `.env` or API keys.
- Generated files under `backend/interview_agent/artifacts/` are runtime outputs and should stay out of version control.
- Large local model/cache files should remain ignored.

## Troubleshooting

- PyAudio build errors on macOS: `brew install portaudio` then reinstall voice requirements.
- Missing API key errors: verify `.env` values and active shell environment.
- Webcam/emotion issues: check camera permissions and OpenCV install.
- Voice session disconnects: validate Cartesia/ElevenLabs credentials and agent IDs.

## License

Add a `LICENSE` file before publishing if this repository will be public.
