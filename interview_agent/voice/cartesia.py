"""
Cartesia Line agent WebSocket session (Calls API).

Refactored from cartesia_agent_voice.py for use as a library and LangGraph node.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

CARTESIA_VERSION = "2025-04-16"
ACCESS_TOKEN_URL = "https://api.cartesia.ai/access-token"
WS_TEMPLATE = "wss://api.cartesia.ai/agents/stream/{agent_id}"
CALLS_URL = "https://api.cartesia.ai/agents/calls"
STT_URL = "https://api.cartesia.ai/stt"

INPUT_FORMATS = ("pcm_44100", "pcm_24000", "pcm_16000")


class CartesiaConfigError(RuntimeError):
    pass


def _is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError | socket.timeout):
            return True
        text = str(reason or exc).lower()
        return any(
            token in text
            for token in (
                "unexpected_eof_while_reading",
                "eof occurred in violation of protocol",
                "timed out",
                "connection reset",
                "temporarily unavailable",
                "tlsv1 alert",
            )
        )
    return False


def _urlopen_json_with_retry(
    req: urllib.request.Request,
    *,
    timeout: int,
    retries: int = 3,
    base_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # network/TLS/transient transport failures
            last_error = exc
            if attempt >= retries or not _is_retryable_url_error(exc):
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"[Cartesia] transient network error (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {delay:.1f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise CartesiaConfigError(f"Missing or empty environment variable: {name}")
    return value


def load_questions_file(path: str) -> list[str]:
    text = Path(path).expanduser().read_text(encoding="utf-8")
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def mint_agent_access_token(api_key: str, expires_in: int) -> str:
    body = json.dumps({"grants": {"agent": True}, "expires_in": expires_in}).encode()
    req = urllib.request.Request(
        ACCESS_TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Cartesia-Version": CARTESIA_VERSION,
        },
    )
    try:
        payload = _urlopen_json_with_retry(req, timeout=60, retries=3)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise CartesiaConfigError(
            f"access-token request failed ({exc.code}): {detail}"
        ) from exc
    except Exception as exc:
        raise CartesiaConfigError(
            "access-token request failed due to a network/TLS error: "
            f"{exc}. You can retry, set CARTESIA_ACCESS_TOKEN, or run with direct API key auth."
        ) from exc
    token = payload.get("token")
    if not token:
        raise CartesiaConfigError("access-token response missing 'token' field.")
    return token


def resolve_bearer_token(*, use_api_key_directly: bool, expires_in: int) -> str:
    env_token = os.getenv("CARTESIA_ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token
    api_key = require_env("CARTESIA_API_KEY")
    if use_api_key_directly:
        return api_key
    try:
        return mint_agent_access_token(api_key, expires_in=expires_in)
    except CartesiaConfigError as exc:
        # Fallback for intermittent /access-token transport failures.
        print(
            f"[Cartesia] Falling back to direct API key bearer after token mint failure: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return api_key


def _pcm_sample_rate(input_format: str) -> int:
    return int(input_format.rsplit("_", 1)[-1])


def _sample_width_for_encoding(encoding: str) -> int:
    if encoding == "pcm_f32le":
        return 4
    if encoding == "pcm_s16le":
        return 2
    raise ValueError(f"unknown encoding {encoding}")


def _mean_abs_pcm16(frame: bytes) -> float:
    """Cheap signal level estimate for silence gating."""
    if not frame:
        return 0.0
    samples = memoryview(frame).cast("h")
    return sum(abs(s) for s in samples) / len(samples)


def _collect_text_values(obj: Any, *, _key: str = "") -> list[str]:
    """Best-effort extraction of textual snippets from websocket events."""
    out: list[str] = []
    skip_keys = {"payload", "media", "audio", "blob", "bytes", "pcm"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in skip_keys:
                continue
            out.extend(_collect_text_values(v, _key=str(k)))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_text_values(item, _key=_key))
    elif isinstance(obj, str):
        s = obj.strip()
        if s and len(s) <= 2000:
            out.append(s)
    return out


def _speaker_for_event(event_name: str) -> str:
    e = (event_name or "").lower()
    if "agent" in e or "assistant" in e or "output" in e:
        return "Agent"
    if "user" in e or "candidate" in e or "input" in e:
        return "Candidate"
    return "Event"


def _save_session_artifacts(
    *,
    transcript_lines: list[str],
    event_log: list[dict[str, Any]],
    mic_pcm: bytes,
    mic_sample_rate: int,
) -> tuple[str, str, str | None]:
    base_dir = Path("interview_agent/artifacts/sessions").resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_dir / "transcript.md"
    event_log_path = run_dir / "events.jsonl"
    mic_audio_path = run_dir / "mic_input.wav"

    transcript_body = "# Interview Transcript\n\n"
    if transcript_lines:
        transcript_body += "\n".join(f"- {line}" for line in transcript_lines) + "\n"
    else:
        transcript_body += "_No transcript text events were captured from Cartesia._\n"
    transcript_path.write_text(transcript_body, encoding="utf-8")

    with event_log_path.open("w", encoding="utf-8") as fh:
        for event in event_log:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    saved_audio: str | None = None
    if mic_pcm:
        with wave.open(str(mic_audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(mic_sample_rate)
            wav.writeframes(mic_pcm)
        saved_audio = str(mic_audio_path)

    return str(transcript_path), str(event_log_path), saved_audio


def build_start_agent_and_metadata(
    questions: list[str],
    *,
    introduction: str | None,
    system_prompt_file: str | Path | None,
    questions_metadata_only: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Mirror CLI behavior: session_questions in metadata + optional system_prompt appendix."""
    start_metadata: dict[str, Any] = dict(extra_metadata or {})
    start_agent: dict[str, Any] = {}
    cleaned = [q.strip() for q in questions if q and str(q).strip()]
    if not cleaned:
        return None, None

    start_metadata["session_questions"] = cleaned
    appendix = (
        "\n\nSession script — ask the user these questions in order. "
        "Ask one at a time and acknowledge their answers before continuing. "
        "After the final answer, thank the candidate and end the call. "
        "If your runtime exposes an end_call tool or AgentEndCall, use it immediately after the closing message:\n"
        + "\n".join(f"{i}. {q}" for i, q in enumerate(cleaned, 1))
    )

    if introduction:
        start_agent["introduction"] = introduction

    if system_prompt_file:
        base = Path(system_prompt_file).expanduser().read_text(encoding="utf-8")
        if questions_metadata_only:
            start_agent["system_prompt"] = base.rstrip()
        else:
            start_agent["system_prompt"] = base.rstrip() + appendix
    elif appendix and not questions_metadata_only:
        start_agent["system_prompt"] = (
            "Follow your configured agent role and speaking style.\n" + appendix.strip()
        )

    return start_agent or None, start_metadata or None


async def _run_stream(
    agent_id: str,
    bearer: str,
    input_format: str,
    *,
    agent_output_encoding: str,
    output_sample_rate: int,
    start_agent: dict[str, Any] | None,
    start_metadata: dict[str, Any] | None,
    session_id: str | None = None,
    mic_gate_level: float = 0.0,
    suppress_mic_ms_after_playback: int = 0,
    manual_stop: bool = False,
    auto_stop_after_silence_seconds: float = 45.0,
    max_session_seconds: float = 900.0,
    stop_signal_file: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import pyaudio
    except ImportError as exc:
        raise CartesiaConfigError(
            "PyAudio is required for the voice interview. On macOS: "
            "brew install portaudio && pip install pyaudio"
        ) from exc

    import websockets
    import websockets.exceptions

    uri = WS_TEMPLATE.format(agent_id=agent_id)
    headers = [
        ("Authorization", f"Bearer {bearer}"),
        ("Cartesia-Version", CARTESIA_VERSION),
    ]

    mic_rate = _pcm_sample_rate(input_format)
    chunk_samples = max(int(mic_rate * 0.02), 1)
    sample_width = _sample_width_for_encoding(agent_output_encoding)

    pa = pyaudio.PyAudio()
    mic_stream = None
    out_stream = None

    stop = asyncio.Event()
    event_log: list[dict[str, Any]] = []
    transcript_lines: list[str] = []
    mic_pcm = bytearray()

    try:
        print("[Cartesia] Connecting to voice agent websocket...", flush=True)
        async with websockets.connect(
            uri,
            additional_headers=headers,
            max_size=None,
        ) as ws:
            start_payload: dict[str, Any] = {
                "event": "start",
                "config": {"input_format": input_format},
            }
            if session_id:
                start_payload["stream_id"] = session_id
            if start_metadata:
                start_payload["metadata"] = start_metadata
            if start_agent:
                start_payload["agent"] = start_agent
            await ws.send(json.dumps(start_payload))
            print("[Cartesia] Start event sent; waiting for session ack...", flush=True)

            stream_id: str | None = None
            while stream_id is None:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                msg = json.loads(raw)
                event_log.append(msg)
                ev = msg.get("event")
                if ev == "ack":
                    stream_id = msg.get("stream_id")
                    cfg = msg.get("config")
                    agent_cfg = msg.get("agent") or {}
                    print("[Cartesia] Session ack — stream_id:", stream_id, flush=True)
                    if cfg:
                        print("[Cartesia] Config:", json.dumps(cfg, indent=2), flush=True)
                    intro = agent_cfg.get("introduction")
                    if intro:
                        print("[Cartesia] Introduction (from server):", intro, flush=True)
                elif ev == "error":
                    print("[Cartesia] Server error:", json.dumps(msg), file=sys.stderr, flush=True)
                    return {}
                else:
                    print("(waiting for ack, got)", ev, json.dumps(msg)[:300])

            assert stream_id is not None

            mic_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=mic_rate,
                input=True,
                frames_per_buffer=chunk_samples,
            )
            out_fmt = (
                pyaudio.paFloat32
                if agent_output_encoding == "pcm_f32le"
                else pyaudio.paInt16
            )
            out_chunk = max(int(output_sample_rate * 0.02), 1)
            out_stream = pa.open(
                format=out_fmt,
                channels=1,
                rate=output_sample_rate,
                output=True,
                frames_per_buffer=out_chunk * 2,
            )
            out_pcm_remainder = bytearray()
            loop = asyncio.get_running_loop()
            started_ts = loop.time()
            last_playback_ts = started_ts
            last_input_activity_ts = started_ts

            print(
                f"Mic: pcm_s16le @ {mic_rate} Hz  →  Agent playback: {agent_output_encoding} @ "
                f"{output_sample_rate} Hz\nIf audio sounds wrong, try output_sample_rate=24000 "
                "or agent_output_encoding=pcm_f32le.\n"
                "[Cartesia] Live interview is running. Talk normally.\n",
                flush=True,
            )
            if manual_stop:
                print("[Cartesia] Enter-stop enabled: press Enter when complete.", flush=True)
            else:
                print(
                    "[Cartesia] Auto-stop enabled: the session ends when the agent closes "
                    f"or after {auto_stop_after_silence_seconds:g}s of silence. "
                    "Press Enter at any time to end immediately.",
                    flush=True,
                )
            if mic_gate_level > 0.0 or suppress_mic_ms_after_playback > 0:
                print(
                    "Mic guardrails:"
                    f" gate={mic_gate_level:g},"
                    f" suppress_after_playback={suppress_mic_ms_after_playback}ms\n"
                )

            async def send_mic() -> None:
                nonlocal last_input_activity_ts
                assert mic_stream is not None
                while not stop.is_set():
                    try:
                        pcm = await asyncio.to_thread(
                            mic_stream.read, chunk_samples, False
                        )
                    except OSError:
                        if stop.is_set():
                            break
                        raise

                    level = _mean_abs_pcm16(pcm)
                    if level > max(mic_gate_level, 200.0):
                        last_input_activity_ts = asyncio.get_running_loop().time()

                    # Ignore likely speaker bleed while agent audio is actively playing.
                    if suppress_mic_ms_after_playback > 0:
                        delta_ms = (
                            asyncio.get_running_loop().time() - last_playback_ts
                        ) * 1000.0
                        if delta_ms < suppress_mic_ms_after_playback:
                            continue

                    # Drop low-level ambient noise so VAD doesn't keep interrupting.
                    if mic_gate_level > 0.0 and level < mic_gate_level:
                        continue

                    mic_pcm.extend(pcm)
                    b64 = base64.b64encode(pcm).decode("ascii")
                    await ws.send(
                        json.dumps(
                            {
                                "event": "media_input",
                                "stream_id": stream_id,
                                "media": {"payload": b64},
                            }
                        )
                    )

            async def recv_loop() -> None:
                nonlocal last_playback_ts
                assert out_stream is not None
                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed as closed:
                        print(
                            f"\nWebSocket closed: {closed.code} {closed.reason}\n",
                            file=sys.stderr,
                        )
                        stop.set()
                        break
                    msg: dict[str, Any] = json.loads(raw)
                    event_log.append(msg)
                    ev = msg.get("event")
                    if ev != "media_output":
                        snippets = _collect_text_values(msg)
                        if snippets:
                            speaker = _speaker_for_event(str(ev))
                            joined = " | ".join(snippets[:3])
                            transcript_lines.append(f"{speaker}: {joined}")
                    if ev == "media_output":
                        blob = base64.b64decode(msg["media"]["payload"])
                        last_playback_ts = asyncio.get_running_loop().time()
                        out_pcm_remainder.extend(blob)
                        aligned_len = len(out_pcm_remainder) - (
                            len(out_pcm_remainder) % sample_width
                        )
                        if aligned_len > 0:
                            to_play = bytes(out_pcm_remainder[:aligned_len])
                            del out_pcm_remainder[:aligned_len]
                            await asyncio.to_thread(out_stream.write, to_play)
                    elif ev == "clear":
                        out_pcm_remainder.clear()
                        print("\n(agent requested playback clear / interrupt)\n")
                    elif ev == "transfer_call":
                        target = (msg.get("transfer") or {}).get("target_phone_number")
                        print(f"\n(transfer_call requested to {target})\n")
                    elif ev in ("ack", "ping", "pong"):
                        pass
                    else:
                        print("\n[event]", ev, json.dumps(msg)[:400])

            async def keepalive() -> None:
                while not stop.is_set():
                    await asyncio.sleep(55)
                    try:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=15)
                    except Exception:
                        break

            async def wait_stop() -> None:
                await asyncio.to_thread(
                    input,
                    "\n[Cartesia] Press Enter when the interview is finished. "
                    "This ends voice capture and starts transcript/report processing...\n",
                )
                print("[Cartesia] Stop requested by user; closing voice session...", flush=True)
                stop.set()

            async def wait_stop_signal_file() -> None:
                if not stop_signal_file:
                    return
                signal_path = Path(stop_signal_file).expanduser().resolve()
                while not stop.is_set():
                    if signal_path.exists():
                        print(
                            "[Cartesia] Stop signal received from UI; closing voice session...",
                            flush=True,
                        )
                        stop.set()
                        break
                    await asyncio.sleep(0.25)

            async def auto_stop() -> None:
                while not stop.is_set():
                    await asyncio.sleep(1)
                    now = asyncio.get_running_loop().time()
                    if now - started_ts >= max_session_seconds:
                        print("[Cartesia] Max session duration reached; closing voice session...", flush=True)
                        stop.set()
                        break
                    last_activity = max(last_playback_ts, last_input_activity_ts)
                    if now - last_activity >= auto_stop_after_silence_seconds:
                        print(
                            "[Cartesia] Auto-stop silence timeout reached; closing voice session...",
                            flush=True,
                        )
                        stop.set()
                        break

            send_task = asyncio.create_task(send_mic())
            recv_task = asyncio.create_task(recv_loop())
            ping_task = asyncio.create_task(keepalive())
            stop_tasks: set[asyncio.Task[Any]] = set()
            if stop_signal_file:
                stop_tasks.add(asyncio.create_task(wait_stop_signal_file()))
            else:
                stop_tasks.add(asyncio.create_task(wait_stop()))
            if not manual_stop:
                stop_tasks.add(asyncio.create_task(auto_stop()))

            done, pending = await asyncio.wait(
                {send_task, recv_task, ping_task, *stop_tasks},
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop.set()
            for t in pending:
                t.cancel()
            for t in done:
                if not t.cancelled() and (exc := t.exception()):
                    print("Task error:", exc, file=sys.stderr)
            await asyncio.gather(send_task, recv_task, ping_task, *stop_tasks, return_exceptions=True)

            await ws.close(code=1000, reason="client finished")
            print("[Cartesia] Voice session closed. Continuing graph...", flush=True)

    except websockets.exceptions.InvalidStatus as exc:
        raise CartesiaConfigError(f"WebSocket handshake failed: {exc}") from exc
    finally:
        if mic_stream is not None:
            mic_stream.stop_stream()
            mic_stream.close()
        if out_stream is not None:
            out_stream.stop_stream()
            out_stream.close()
        pa.terminate()
    transcript_path, event_log_path, mic_audio_path = _save_session_artifacts(
        transcript_lines=transcript_lines,
        event_log=event_log,
        mic_pcm=bytes(mic_pcm),
        mic_sample_rate=mic_rate,
    )
    return {
        "transcript_lines": transcript_lines,
        "transcript_path": transcript_path,
        "event_log_path": event_log_path,
        "mic_audio_path": mic_audio_path,
    }


def run_voice_interview_sync(
    questions: list[str],
    *,
    agent_id: str | None = None,
    input_format: str = "pcm_44100",
    use_api_key_directly: bool = False,
    expires_in: int = 3600,
    output_sample_rate: int | None = None,
    agent_output_float: bool = False,
    mic_gate_level: float = 0.0,
    suppress_mic_ms_after_playback: int = 0,
    introduction: str | None = None,
    system_prompt_file: str | Path | None = None,
    questions_metadata_only: bool = False,
    session_id: str | None = None,
    manual_stop: bool = False,
    auto_stop_after_silence_seconds: float | None = None,
    max_session_seconds: float | None = None,
    stop_signal_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run a blocking Cartesia voice session with the given questions.

    Uses CARTESIA_AGENT_ID and CARTESIA_API_KEY (or CARTESIA_ACCESS_TOKEN) from the environment.
    """
    aid = (agent_id or os.getenv("CARTESIA_AGENT_ID", "")).strip()
    if not aid:
        raise CartesiaConfigError("Set CARTESIA_AGENT_ID or pass agent_id=")

    if not os.getenv("CARTESIA_ACCESS_TOKEN", "").strip():
        require_env("CARTESIA_API_KEY")

    bearer = resolve_bearer_token(
        use_api_key_directly=use_api_key_directly,
        expires_in=expires_in,
    )

    agent_encoding = "pcm_f32le" if agent_output_float else "pcm_s16le"
    default_sr = 44100
    env_sr = os.getenv("CARTESIA_OUTPUT_SAMPLE_RATE", "").strip()
    if env_sr.isdigit():
        default_sr = int(env_sr)
    sr = output_sample_rate if output_sample_rate is not None else default_sr
    session_started_at = datetime.now(timezone.utc).isoformat()
    auto_silence = (
        auto_stop_after_silence_seconds
        if auto_stop_after_silence_seconds is not None
        else float(os.getenv("CARTESIA_AUTO_STOP_AFTER_SILENCE_SECONDS", "45"))
    )
    max_duration = (
        max_session_seconds
        if max_session_seconds is not None
        else float(os.getenv("CARTESIA_MAX_SESSION_SECONDS", "900"))
    )

    start_agent, start_metadata = build_start_agent_and_metadata(
        questions,
        introduction=introduction,
        system_prompt_file=system_prompt_file,
        questions_metadata_only=questions_metadata_only,
        extra_metadata={
            "session_id": session_id,
            "stream_id": session_id,
            "session_started_at": session_started_at,
            "source": "interview_agent",
        }
        if session_id
        else {"session_started_at": session_started_at, "source": "interview_agent"},
    )

    print(
        "Voice interview session questions:",
        json.dumps(
            [
                q
                for q in (start_metadata or {}).get("session_questions", [])
                or questions
            ],
            indent=2,
            ensure_ascii=False,
        ),
    )

    result = asyncio.run(
        _run_stream(
            aid,
            bearer,
            input_format,
            agent_output_encoding=agent_encoding,
            output_sample_rate=sr,
            start_agent=start_agent,
            start_metadata=start_metadata,
            session_id=session_id,
            mic_gate_level=mic_gate_level,
            suppress_mic_ms_after_playback=suppress_mic_ms_after_playback,
            manual_stop=manual_stop,
            auto_stop_after_silence_seconds=auto_silence,
            max_session_seconds=max_duration,
            stop_signal_file=stop_signal_file,
        )
    )
    result["agent_id"] = aid
    result["session_id"] = session_id
    result["session_started_at"] = session_started_at
    return result


def _call_cartesia_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    api_key = require_env("CARTESIA_API_KEY")
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": CARTESIA_VERSION,
        },
    )
    try:
        return _urlopen_json_with_retry(req, timeout=timeout, retries=3)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise CartesiaConfigError(
            f"Cartesia Calls API request failed ({exc.code}): {detail}"
        ) from exc
    except Exception as exc:
        raise CartesiaConfigError(
            f"Cartesia Calls API request failed due to network/TLS error: {exc}"
        ) from exc


def _post_multipart_cartesia_stt(
    *,
    audio_path: str | Path,
    model: str = "ink-whisper",
    language: str = "en",
    timeout: int = 180,
) -> dict[str, Any]:
    api_key = require_env("CARTESIA_API_KEY")
    path = Path(audio_path).expanduser().resolve()
    boundary = f"----interview-agent-{int(time.time() * 1000)}"

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    body = bytearray()
    body.extend(field("model", model))
    body.extend(field("language", language))
    body.extend(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        STT_URL,
        data=bytes(body),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        return _urlopen_json_with_retry(req, timeout=timeout, retries=3)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise CartesiaConfigError(
            f"Cartesia STT request failed ({exc.code}): {detail}"
        ) from exc
    except Exception as exc:
        raise CartesiaConfigError(
            f"Cartesia STT request failed due to network/TLS error: {exc}"
        ) from exc


def transcribe_mic_audio_fallback(
    *,
    audio_path: str | Path,
    session_id: str,
) -> dict[str, Any]:
    print("[Cartesia] Running STT fallback on recorded mic audio...", flush=True)
    payload = _post_multipart_cartesia_stt(audio_path=audio_path)
    text = str(payload.get("text") or "").strip()

    base_dir = Path("interview_agent/artifacts/sessions").resolve()
    run_dir = base_dir / f"{session_id}_stt_fallback"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "cartesia_stt.json"
    transcript_path = run_dir / "stt_fallback_transcript.md"

    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript_body = "# Interview Transcript (STT Fallback)\n\n"
    if text:
        transcript_body += f"**Candidate:** {text}\n"
    else:
        transcript_body += "_Cartesia STT returned no candidate speech text._\n"
    transcript_path.write_text(transcript_body, encoding="utf-8")

    return {
        "transcript_path": str(transcript_path),
        "raw_stt_path": str(raw_path),
        "transcript_text": transcript_body,
    }


def _call_matches_session(call: dict[str, Any], session_id: str) -> bool:
    metadata = call.get("metadata") or {}
    if metadata.get("session_id") == session_id:
        return True
    if metadata.get("stream_id") == session_id:
        return True
    if call.get("stream_id") == session_id:
        return True
    telephony_params = call.get("telephony_params") or {}
    params = telephony_params.get("parameters") or {}
    if params.get("session_id") == session_id or params.get("stream_id") == session_id:
        return True
    return telephony_params.get("call_sid") == session_id


def _parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _is_recent_websocket_call(
    call: dict[str, Any],
    *,
    agent_id: str,
    started_after: datetime | None,
) -> bool:
    if call.get("agent_id") != agent_id:
        return False
    telephony_params = call.get("telephony_params") or {}
    if telephony_params.get("connection_type") not in (None, "websocket"):
        return False
    if started_after is None:
        return True
    call_start = _parse_rfc3339(call.get("start_time"))
    if call_start is None:
        return True
    return call_start >= started_after


def _format_official_transcript(transcript: list[dict[str, Any]]) -> str:
    lines = ["# Interview Transcript", ""]
    for turn in transcript:
        role = str(turn.get("role") or "unknown").title()
        text = str(turn.get("text") or "").strip()
        if not text:
            chunks = turn.get("text_chunks") or []
            text = "".join(str(chunk.get("text") or "") for chunk in chunks).strip()
        start = turn.get("start_timestamp")
        end = turn.get("end_timestamp")
        stamp = ""
        if start is not None and end is not None:
            stamp = f" [{start:.2f}s-{end:.2f}s]" if isinstance(start, (int, float)) and isinstance(end, (int, float)) else f" [{start}-{end}]"
        if text:
            lines.append(f"**{role}{stamp}:** {text}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_official_call_artifacts(call: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    base_dir = Path("interview_agent/artifacts/sessions").resolve()
    call_id = str(call.get("id") or session_id)
    run_dir = base_dir / call_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "cartesia_call.json"
    transcript_path = run_dir / "official_transcript.md"
    raw_path.write_text(json.dumps(call, ensure_ascii=False, indent=2), encoding="utf-8")

    transcript = call.get("transcript") or []
    transcript_text = _format_official_transcript(transcript)
    transcript_path.write_text(transcript_text, encoding="utf-8")

    return {
        "call_id": call_id,
        "transcript_path": str(transcript_path),
        "raw_call_path": str(raw_path),
        "transcript_text": transcript_text,
    }


def _save_calls_debug_payload(payload: dict[str, Any], *, session_id: str) -> str:
    base_dir = Path("interview_agent/artifacts/sessions").resolve()
    run_dir = base_dir / f"{session_id}_calls_debug"
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_path = run_dir / "list_calls_payload.json"
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(debug_path)


def fetch_official_call_transcript(
    *,
    agent_id: str,
    session_id: str,
    session_started_at: str | None = None,
    poll_seconds: int = 90,
    poll_interval_seconds: float = 3.0,
    page_limit: int = 20,
) -> dict[str, Any]:
    """
    Poll Cartesia Calls API for the completed websocket call that carries our metadata.session_id.
    """
    deadline = time.monotonic() + poll_seconds
    query = urllib.parse.urlencode(
        {
            "agent_id": agent_id,
            "expand": "transcript",
            "limit": str(page_limit),
        }
    )
    url = f"{CALLS_URL}?{query}"

    last_seen = None
    started_after = _parse_rfc3339(session_started_at)
    print(
        f"[Cartesia] Polling official transcript for session_id={session_id}...",
        flush=True,
    )
    while True:
        payload = _call_cartesia_json(url)
        calls = payload.get("data") or []
        last_seen = len(calls)
        candidates: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            if _call_matches_session(call, session_id):
                candidates.insert(0, call)
                continue
            if _is_recent_websocket_call(
                call,
                agent_id=agent_id,
                started_after=started_after,
            ):
                candidates.append(call)

        for call in candidates:
            transcript = call.get("transcript") or []
            if transcript:
                print(
                    f"[Cartesia] Official transcript is ready for call {call.get('id')}.",
                    flush=True,
                )
                return save_official_call_artifacts(call, session_id=session_id)
            status = call.get("status")
            if status in {"completed", "failed", "errored"}:
                print(
                    f"[Cartesia] Call {call.get('id')} ended with status={status}; saving available transcript data.",
                    flush=True,
                )
                return save_official_call_artifacts(call, session_id=session_id)

        if time.monotonic() >= deadline:
            debug_path = _save_calls_debug_payload(payload, session_id=session_id)
            raise CartesiaConfigError(
                f"No Cartesia call transcript found for session_id={session_id!r}; "
                f"checked {last_seen or 0} recent calls. Debug payload: {debug_path}"
            )
        print(
            f"[Cartesia] Transcript not ready yet; retrying in {poll_interval_seconds:g}s...",
            flush=True,
        )
        time.sleep(poll_interval_seconds)
