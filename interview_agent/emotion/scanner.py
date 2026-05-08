"""Webcam emotion monitoring for interview sessions."""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EmotionMonitorResult:
    csv_path: str
    summary: str
    sample_count: int
    error: str | None = None


def _load_emotion_stack():
    """Import heavy CV/ML dependencies only when the monitor is actually used."""
    os.environ.setdefault("HF_HOME", "./models")

    import cv2
    import numpy as np
    import torch
    from hsemotion.facial_emotions import HSEmotionRecognizer

    # HSEmotion loads full timm checkpoints; PyTorch 2.6+ defaults to
    # weights_only=True, which breaks these checkpoints.
    orig_torch_load = torch.load

    def torch_load_hsemotion(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    torch.load = torch_load_hsemotion
    return cv2, np, torch, HSEmotionRecognizer


def _emotion_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _open_camera(cv2_module: Any, preferred_index: int):
    candidate_indices: list[int] = []
    for idx in [preferred_index, 0, 1, 2, 3]:
        if idx not in candidate_indices:
            candidate_indices.append(idx)

    for idx in candidate_indices:
        cap = cv2_module.VideoCapture(idx, cv2_module.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap, idx
        cap.release()
        cap = cv2_module.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()
    raise RuntimeError(
        f"Could not open webcam. Tried camera indexes: {candidate_indices}"
    )


def summarize_emotion_csv(path: str | Path) -> EmotionMonitorResult:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return EmotionMonitorResult(
            csv_path=str(csv_path),
            summary="Emotion monitoring did not produce a CSV file.",
            sample_count=0,
            error="CSV missing",
        )

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return EmotionMonitorResult(
            csv_path=str(csv_path),
            summary="Emotion monitoring ran but captured no face/emotion samples.",
            sample_count=0,
        )

    emotions = [
        r.get("dominant_emotion", "").strip()
        for r in rows
        if r.get("dominant_emotion")
    ]
    counts = Counter(emotions)
    confidences: list[float] = []
    timestamps: list[float] = []
    for row in rows:
        try:
            confidences.append(float(row.get("confidence") or 0.0))
        except ValueError:
            pass
        try:
            timestamps.append(float(row.get("timestamp") or 0.0))
        except ValueError:
            pass

    duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    no_face = counts.get("no_face", 0)
    face_counts = Counter({k: v for k, v in counts.items() if k != "no_face"})
    total = max(sum(face_counts.values()), 1)
    distribution = ", ".join(
        f"{emotion}: {count} ({(count / total) * 100:.1f}%)"
        for emotion, count in face_counts.most_common()
    )
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    dominant = face_counts.most_common(1)[0][0] if face_counts else "no_face_detected"

    summary = (
        f"Emotion monitor captured {len(rows)} samples over {duration:.1f}s. "
        f"Face-detected samples: {sum(face_counts.values())}. "
        f"No-face samples: {no_face}. "
        f"Most common emotion: {dominant}. "
        f"Distribution: {distribution or 'n/a'}. "
        f"Average model confidence: {avg_conf:.2f}."
    )
    return EmotionMonitorResult(
        csv_path=str(csv_path),
        summary=summary,
        sample_count=len(rows),
    )


def run_emotion_scanner(
    *,
    csv_path: str | Path,
    stop_event: threading.Event | None = None,
    camera_index: int = 0,
    show_window: bool = False,
    model_name: str | None = None,
    preview_frame_path: str | Path | None = None,
) -> EmotionMonitorResult:
    """
    Run webcam emotion detection until stop_event is set.

    This mirrors the original `emotion_scanner.py`, but is importable and
    lifecycle-controlled by the interview graph.
    """
    stop = stop_event or threading.Event()
    out_path = Path(csv_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "dominant_emotion", "confidence"])

    model_stack_error: str | None = None
    emotion_classifier = None
    np = None
    try:
        cv2, np, torch, recognizer_cls = _load_emotion_stack()
        chosen_model = model_name or os.environ.get(
            "HSEMOTION_MODEL", "enet_b0_8_best_afew"
        )
        device = _emotion_device(torch)
        print(f"Loading HSEmotion model {chosen_model} for interview monitor...")
        emotion_classifier = recognizer_cls(model_name=chosen_model, device=device)
        print(f"HSEmotion loaded on {device}.")
    except Exception as exc:
        model_stack_error = str(exc)
        import cv2  # type: ignore

        print(
            "Emotion model stack unavailable; falling back to camera-only capture:"
            f" {model_stack_error}"
        )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap, active_camera_index = _open_camera(cv2, camera_index)
    print(f"Emotion monitor camera index in use: {active_camera_index}")
    preview_path = (
        Path(preview_frame_path).expanduser().resolve() if preview_frame_path else None
    )
    last_preview_write = 0.0
    last_no_face_write = 0.0
    failed_reads = 0

    try:
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads >= 30:
                    raise RuntimeError("Camera read failed repeatedly; no frames captured.")
                time.sleep(0.05)
                continue
            failed_reads = 0

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                )
                has_face = len(faces) > 0
                wrote_row = False

                for (x, y, w, h) in faces:
                    ih, iw, _ = frame.shape
                    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(iw, x + w + pad_x)
                    y2 = min(ih, y + h + pad_y)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_roi = rgb_frame[y1:y2, x1:x2]

                    if emotion_classifier is not None and np is not None:
                        emotion, scores = emotion_classifier.predict_emotions(
                            face_roi, logits=False
                        )
                        scores_arr = np.asarray(scores).reshape(-1)
                        confidence = float(np.max(scores_arr))
                    else:
                        emotion = "face_detected"
                        confidence = 0.0

                    with out_path.open("a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow([time.time(), emotion, confidence])
                    wrote_row = True

                    if show_window:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{emotion} ({confidence:.2f})"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                if not has_face:
                    now = time.time()
                    if now - last_no_face_write >= 1.0:
                        with out_path.open("a", newline="", encoding="utf-8") as file:
                            writer = csv.writer(file)
                            writer.writerow([now, "no_face", 0.0])
                        last_no_face_write = now
                        wrote_row = True

                if not wrote_row:
                    # Ensure we always leave trace samples while camera is active.
                    now = time.time()
                    if now - last_no_face_write >= 1.0:
                        with out_path.open("a", newline="", encoding="utf-8") as file:
                            writer = csv.writer(file)
                            writer.writerow([now, "camera_active", 0.0])
                        last_no_face_write = now

                if preview_path:
                    now = time.time()
                    if now - last_preview_write >= 0.25:
                        preview_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_preview = preview_path.with_suffix(".tmp.jpg")
                        cv2.imwrite(str(tmp_preview), frame)
                        os.replace(tmp_preview, preview_path)
                        last_preview_write = now

            except Exception as exc:
                print("Emotion monitor frame error:", exc)

            if show_window:
                cv2.imshow("Interview Emotion Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
                    break

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    result = summarize_emotion_csv(out_path)
    if model_stack_error:
        result.error = (
            f"Emotion model fallback mode: {model_stack_error}"
            if not result.error
            else f"{result.error}; Emotion model fallback mode: {model_stack_error}"
        )
        result.summary = f"{result.summary} Running in camera-only fallback mode."
    return result


class BackgroundEmotionMonitor:
    def __init__(
        self,
        *,
        csv_path: str | Path,
        camera_index: int = 0,
        show_window: bool = False,
        model_name: str | None = None,
        preview_frame_path: str | Path | None = None,
    ) -> None:
        self.csv_path = str(Path(csv_path).expanduser().resolve())
        self.camera_index = camera_index
        self.show_window = show_window
        self.model_name = model_name
        self.preview_frame_path = (
            str(Path(preview_frame_path).expanduser().resolve())
            if preview_frame_path
            else None
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: EmotionMonitorResult | None = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def target() -> None:
            try:
                self._result = run_emotion_scanner(
                    csv_path=self.csv_path,
                    stop_event=self._stop,
                    camera_index=self.camera_index,
                    show_window=self.show_window,
                    model_name=self.model_name,
                    preview_frame_path=self.preview_frame_path,
                )
            except Exception as exc:
                self._error = str(exc)
                self._result = summarize_emotion_csv(self.csv_path)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 10.0) -> EmotionMonitorResult:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

        result = self._result or summarize_emotion_csv(self.csv_path)
        if self._error and not result.error:
            result.error = self._error
            result.summary = f"{result.summary} Monitor error: {self._error}"
        return result
