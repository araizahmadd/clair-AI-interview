"""CLI for standalone Cartesia voice sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from interview_agent.voice.cartesia import (
    INPUT_FORMATS,
    CartesiaConfigError,
    _run_stream,
    build_start_agent_and_metadata,
    load_questions_file,
    resolve_bearer_token,
    require_env,
)


def main() -> None:
    load_dotenv()

    default_output_sr = 44100
    _sr_env = os.getenv("CARTESIA_OUTPUT_SAMPLE_RATE", "").strip()
    if _sr_env.isdigit():
        default_output_sr = int(_sr_env)

    parser = argparse.ArgumentParser(
        description=(
            "Talk to a Cartesia Line agent over the Calls API WebSocket "
            "(microphone + speakers)."
        )
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("CARTESIA_AGENT_ID", "").strip(),
        help="Agent ID (default: CARTESIA_AGENT_ID).",
    )
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default="pcm_44100",
        help="Must match how you capture/send PCM from the microphone.",
    )
    parser.add_argument(
        "--use-api-key-directly",
        action="store_true",
        help=(
            "Send CARTESIA_API_KEY as the WebSocket Bearer token without calling "
            "/access-token (default: mint a token with grants.agent)."
        ),
    )
    parser.add_argument(
        "--expires-in",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Lifetime for minted access tokens (default: 3600).",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=default_output_sr,
        metavar="HZ",
        help=(
            "Sample rate for agent audio playback (default: 44100). "
            "Try 24000 if speech sounds wrong."
        ),
    )
    parser.add_argument(
        "--agent-output-float",
        action="store_true",
        help=(
            "Decode agent audio as pcm_f32le (32-bit float). Default is pcm_s16le."
        ),
    )
    parser.add_argument(
        "--mic-gate-level",
        type=float,
        default=float(os.getenv("CARTESIA_MIC_GATE_LEVEL", "0")),
        metavar="LEVEL",
        help=(
            "Drop mic frames below this mean-abs PCM16 level (0 disables). "
            "Useful to avoid false VAD interrupts; try 200-800."
        ),
    )
    parser.add_argument(
        "--suppress-mic-ms-after-playback",
        type=int,
        default=int(os.getenv("CARTESIA_SUPPRESS_MIC_MS", "0")),
        metavar="MS",
        help=(
            "Temporarily mute mic capture right after agent audio output "
            "(helps speaker bleed/echo). Try 200-600."
        ),
    )
    parser.add_argument(
        "--question",
        "-q",
        action="append",
        dest="questions",
        default=[],
        metavar="TEXT",
        help="Question for the agent to ask (repeat -q for multiple).",
    )
    parser.add_argument(
        "--questions-file",
        metavar="PATH",
        help="Text file with one question per line (# starts a comment).",
    )
    parser.add_argument(
        "--introduction",
        metavar="TEXT",
        help="Override agent.introduction for this session only.",
    )
    parser.add_argument(
        "--system-prompt-file",
        metavar="PATH",
        help="UTF-8 file → agent.system_prompt for this session (questions appended unless --questions-metadata-only).",
    )
    parser.add_argument(
        "--metadata-json",
        metavar="JSON",
        help='Extra JSON merged into start event "metadata".',
    )
    parser.add_argument(
        "--questions-metadata-only",
        action="store_true",
        help="Put questions only in metadata.session_questions.",
    )
    args = parser.parse_args()

    if not args.agent_id:
        print("Set CARTESIA_AGENT_ID or pass --agent-id.", file=sys.stderr)
        raise SystemExit(1)

    try:
        if not os.getenv("CARTESIA_ACCESS_TOKEN", "").strip():
            require_env("CARTESIA_API_KEY")
    except CartesiaConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        bearer = resolve_bearer_token(
            use_api_key_directly=args.use_api_key_directly,
            expires_in=args.expires_in,
        )
    except CartesiaConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    agent_encoding = "pcm_f32le" if args.agent_output_float else "pcm_s16le"

    questions: list[str] = list(args.questions or [])
    if args.questions_file:
        questions.extend(load_questions_file(args.questions_file))

    start_metadata: dict[str, Any] = {}
    if args.metadata_json:
        try:
            parsed = json.loads(args.metadata_json)
        except json.JSONDecodeError as exc:
            print("--metadata-json must be valid JSON.", file=sys.stderr)
            raise SystemExit(1) from exc
        if not isinstance(parsed, dict):
            print("--metadata-json must be a JSON object.", file=sys.stderr)
            raise SystemExit(1)
        start_metadata.update(parsed)

    start_agent_override, qa_meta = build_start_agent_and_metadata(
        questions,
        introduction=args.introduction,
        system_prompt_file=args.system_prompt_file,
        questions_metadata_only=args.questions_metadata_only,
    )

    meta_out = {**start_metadata, **(qa_meta or {})} or None
    agent_out = start_agent_override or None

    if questions:
        print(
            "Session questions:",
            json.dumps(questions, indent=2, ensure_ascii=False),
        )
        if args.questions_metadata_only:
            print(
                "(Questions only in metadata.session_questions; wire your Line agent to read them.)"
            )

    import asyncio

    try:
        asyncio.run(
            _run_stream(
                args.agent_id,
                bearer,
                args.input_format,
                agent_output_encoding=agent_encoding,
                output_sample_rate=args.output_sample_rate,
                start_agent=agent_out,
                start_metadata=meta_out,
                mic_gate_level=args.mic_gate_level,
                suppress_mic_ms_after_playback=args.suppress_mic_ms_after_playback,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    except CartesiaConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
