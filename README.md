# Clair 2 - AI Interview Copilot

Clair 2 is a Python application for mock interview preparation and live interview simulation.  
It combines resume parsing, job description analysis, optional company research, AI question generation, voice interview execution, emotion tracking, and report generation.

## What This Repository Includes

- `interview_agent/`: Main LangGraph-based interview workflow.
- `frontend_streamlit.py`: Streamlit UI for end-to-end interview sessions.
- `interview_prep_langgraph.py`: CLI flow for resume/JD to interview questions.
- `cartesia_agent_voice.py`: Cartesia voice session script.
- `elevenlabs_agent_chat.py`: ElevenLabs text/voice interaction script.
- `interview_agent/artifacts/`: Generated transcripts, reports, session logs, and emotion outputs.

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
streamlit run frontend_streamlit.py
```

This launches the full interview flow:
- upload resume PDF
- paste job description
- optionally provide company name
- start interview, then review generated report

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

- `python interview_prep_langgraph.py`  
  Alternate prep flow script.
- `python cartesia_agent_voice.py`  
  Direct Cartesia voice test script.
- `python elevenlabs_agent_chat.py`  
  ElevenLabs conversational script.

## Project Structure (High Level)

```text
.
├── interview_agent/
│   ├── graph/                # LangGraph nodes, state, workflow
│   ├── voice/                # Cartesia voice integration
│   ├── emotion/              # Emotion scanner integration
│   └── artifacts/            # Runtime outputs (reports, logs, transcripts)
├── frontend_streamlit.py     # Web UI
├── cartesia_agent_voice.py   # Cartesia CLI
├── elevenlabs_agent_chat.py  # ElevenLabs CLI
└── requirements-*.txt
```

## Security And Git Hygiene

- Do not commit `.env` or API keys.
- Generated files under `interview_agent/artifacts/` are runtime outputs and should stay out of version control.
- Large local model/cache files should remain ignored.

## Troubleshooting

- PyAudio build errors on macOS: `brew install portaudio` then reinstall voice requirements.
- Missing API key errors: verify `.env` values and active shell environment.
- Webcam/emotion issues: check camera permissions and OpenCV install.
- Voice session disconnects: validate Cartesia/ElevenLabs credentials and agent IDs.

## License

Add a `LICENSE` file before publishing if this repository will be public.
