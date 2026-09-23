"""Explicit OpenAI audio smoke test; the production audio path remains local."""

from __future__ import annotations

import json
import os
import uuid
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.models.schemas import TranscriptSegment
from app.services.audio import _validated_audio_path

TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe-diarize"


def process_audio_cloud(
    audio_path: str, *, allow_upload: bool = False, timeout: float = 300
) -> list[TranscriptSegment]:
    """Upload MP3/WAV only after explicit opt-in and return diarized segments.

    This is a development comparison path. ``process_audio`` stays entirely
    local and never calls this function.
    """

    if not allow_upload:
        raise ValueError("Cloud audio upload requires allow_upload=True")
    path = _validated_audio_path(audio_path)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    request = _transcription_request(path, key)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI transcription returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("OpenAI transcription request failed") from exc

    return _segments_from_diarized_response(payload)


def _transcription_request(path: Path, key: str) -> Request:
    boundary = f"hackalem-{uuid.uuid4().hex}"
    mime = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    fields = {
        "model": TRANSCRIPTION_MODEL,
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    chunks.extend(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8"),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return Request(
        TRANSCRIPTION_URL,
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )


def _segments_from_diarized_response(payload: object) -> list[TranscriptSegment]:
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise RuntimeError("OpenAI did not return diarized segments")

    segments: list[TranscriptSegment] = []
    missing_speaker = False
    for item in payload["segments"]:
        if not isinstance(item, dict):
            raise RuntimeError("OpenAI returned an invalid diarized segment")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = item.get("speaker")
        speaker_id = speaker.strip() if isinstance(speaker, str) else ""
        missing_speaker = missing_speaker or not speaker_id
        segments.append(
            TranscriptSegment(
                id=len(segments) + 1,
                start=float(item["start"]),
                end=float(item["end"]),
                speaker_id=speaker_id or "UNKNOWN",
                text=text.strip(),
            )
        )
    if missing_speaker:
        warnings.warn(
            "OpenAI did not annotate every speaker; missing labels use UNKNOWN.",
            RuntimeWarning,
            stacklevel=2,
        )
    return segments
