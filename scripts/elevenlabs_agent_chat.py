#!/usr/bin/env python3
"""
Interactive CLI for an ElevenLabs Conversational AI agent (text or voice).

Loads ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID from the environment (see .env.example).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"Missing or empty environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _run_text(agent_id: str, starter_messages: Sequence[str]) -> None:
    from elevenlabs import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation

    client = ElevenLabs(api_key=_require_env("ELEVENLABS_API_KEY"))

    def on_agent(text: str) -> None:
        print(f"\nAgent: {text}\n")

    conversation = Conversation(
        client=client,
        agent_id=agent_id,
        requires_auth=True,
        callback_agent_response=on_agent,
    )

    conversation.start_session()
    try:
        for msg in starter_messages:
            print(f"You (starter): {msg}")
            conversation.send_user_message(msg)

        print(
            "Commands: /quit - exit   |   /context <text> - send contextual update (not a user turn)"
        )
        while True:
            line = input("You: ").strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered in ("/quit", "/exit", "quit", "exit"):
                break
            if lowered.startswith("/context "):
                payload = line[len("/context ") :].strip()
                if payload:
                    conversation.send_contextual_update(payload)
                    print("(context update sent)")
                else:
                    print("Usage: /context Your background note here")
                continue
            conversation.send_user_message(line)
    finally:
        conversation.end_session()
        conv_id = conversation.wait_for_session_end()
        if conv_id:
            print(f"Conversation ID: {conv_id}")


def _run_voice(agent_id: str) -> None:
    from elevenlabs import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation

    client = ElevenLabs(api_key=_require_env("ELEVENLABS_API_KEY"))

    try:
        from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

        audio_interface = DefaultAudioInterface()
    except ImportError as exc:
        print(
            "Voice mode needs PyAudio (and on macOS, the PortAudio library from Homebrew).\n"
            "  macOS:\n"
            "    brew install portaudio\n"
            "    pip install -r requirements.txt\n"
            "  Then run again with --mode voice.\n"
            f"  ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)

    def on_agent(text: str) -> None:
        print(f"\nAgent (transcript): {text}\n")

    def on_user(transcript: str) -> None:
        print(f"\nYou (transcript): {transcript}\n")

    conversation = Conversation(
        client=client,
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=audio_interface,
        callback_agent_response=on_agent,
        callback_user_transcript=on_user,
    )

    conversation.start_session()
    try:
        input(
            "Voice session running - use your default mic and speakers.\n"
            "Press Enter here to stop the session...\n"
        )
    finally:
        conversation.end_session()
        conv_id = conversation.wait_for_session_end()
        if conv_id:
            print(f"Conversation ID: {conv_id}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Talk to an ElevenLabs Conversational AI agent from the terminal."
    )
    parser.add_argument(
        "--mode",
        choices=("text", "voice"),
        default="text",
        help="text: type messages (default). voice: microphone + speakers.",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("ELEVENLABS_AGENT_ID", "").strip(),
        help="Agent ID (default: ELEVENLABS_AGENT_ID from the environment).",
    )
    parser.add_argument(
        "--message",
        "-m",
        action="append",
        dest="starter_messages",
        default=[],
        help="Optional starter message(s); can be passed multiple times (text mode only).",
    )
    args = parser.parse_args()

    agent_id = args.agent_id.strip()
    if not agent_id:
        print(
            "No agent ID: set ELEVENLABS_AGENT_ID or pass --agent-id.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "text":
        _run_text(agent_id, args.starter_messages)
    else:
        if args.starter_messages:
            print("--message is ignored in voice mode.", file=sys.stderr)
        _run_voice(agent_id)


if __name__ == "__main__":
    main()
