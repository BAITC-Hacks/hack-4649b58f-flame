"""Local speech-to-text and optional speaker diarization."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.models.schemas import TranscriptSegment

LOGGER = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "mlx-community/whisper-small-mlx-q4"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav"}


@dataclass(frozen=True)
class _SpeakerTurn:
    start: float
    end: float
    speaker_id: str


def process_audio(audio_path: str) -> list[TranscriptSegment]:
    """Transcribe MP3/WAV locally and attach speaker labels when available.

    MLX Whisper performs multilingual transcription on Apple Silicon.  The
    optional pyannote pipeline is also executed locally.  If diarization cannot
    run, the transcript is still returned with ``speaker_id="UNKNOWN"`` and a
    warning containing the reason.
    """

    path = _validated_audio_path(audio_path)
    transcription = _transcribe(path)
    segments = _segments_from_transcription(transcription)
    if not segments:
        return []

    try:
        speaker_turns = _diarize(path)
        if not speaker_turns:
            raise RuntimeError("the diarization pipeline returned no speaker turns")
    except Exception as exc:  # STT must survive every optional diarization failure.
        _warn_diarization_fallback(exc)
        return segments

    return _assign_speakers(segments, speaker_turns)


def _validated_audio_path(audio_path: str) -> Path:
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ValueError(f"Unsupported audio format {path.suffix!r}; expected {supported}")
    return path.resolve()


def _transcribe(path: Path) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ModuleNotFoundError as exc:
        if exc.name != "mlx_whisper":
            raise RuntimeError(f"mlx-whisper could not load a dependency: {exc}") from exc
        raise RuntimeError(
            "mlx-whisper is not installed; run `pip install -r requirements-audio.txt`"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(f"mlx-whisper could not be initialized: {exc}") from exc

    model = os.getenv("AUDIO_STT_MODEL", DEFAULT_STT_MODEL)
    language = os.getenv("AUDIO_LANGUAGE") or None
    LOGGER.info("Transcribing %s locally with %s", path, model)
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    if not isinstance(result, dict):
        raise RuntimeError("mlx-whisper returned an unexpected result")
    return result


def _segments_from_transcription(result: dict[str, Any]) -> list[TranscriptSegment]:
    # A forced language is authoritative.  Automatic file-level detection is
    # propagated only when the backend exposes a high confidence; mixed speech
    # must not be mislabeled from a single low-confidence guess.
    language = os.getenv("AUDIO_LANGUAGE") or None
    if language is None:
        probability = result.get("language_probability")
        if isinstance(probability, (int, float)) and probability >= 0.80:
            detected = result.get("language")
            language = detected if isinstance(detected, str) else None

    transcript: list[TranscriptSegment] = []
    for raw_segment in result.get("segments") or []:
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, float(raw_segment.get("start", 0.0)))
        end = max(start, float(raw_segment.get("end", start)))
        transcript.append(
            TranscriptSegment(
                id=len(transcript) + 1,
                start=start,
                end=end,
                speaker_id="UNKNOWN",
                text=text,
                language=language,
            )
        )
    return transcript


def _diarize(path: Path) -> list[_SpeakerTurn]:
    # This must be set before importing pyannote so its optional metrics remain
    # disabled even during pipeline construction.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "pyannote", "pyannote.audio"}:
            raise RuntimeError(f"pyannote.audio could not load a dependency: {exc}") from exc
        raise RuntimeError(
            "pyannote.audio is not installed; install requirements-audio.txt"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(f"pyannote.audio could not be initialized: {exc}") from exc

    model = os.getenv("AUDIO_DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)
    token = os.getenv("HF_TOKEN") or None
    LOGGER.info("Diarizing %s locally with %s", path, model)
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except TypeError:
        # Compatibility with pyannote 3.x, which used use_auth_token.
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            "could not load the diarization model; accept its Hugging Face terms "
            "and set HF_TOKEN, or point AUDIO_DIARIZATION_MODEL to a local copy"
        )

    device_name = os.getenv("AUDIO_DIARIZATION_DEVICE", "cpu")
    pipeline.to(torch.device(device_name))
    output = pipeline(str(path))
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    turns: list[_SpeakerTurn] = []
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            _SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker_id=_normalise_speaker_id(str(speaker)),
            )
        )
    return turns


def _normalise_speaker_id(label: str) -> str:
    label = label.strip()
    if not label:
        return "UNKNOWN"
    if label.startswith("SPEAKER_"):
        return label
    if label.isdigit():
        return f"SPEAKER_{int(label):02d}"
    return label


def _assign_speakers(
    segments: Iterable[TranscriptSegment], turns: Iterable[_SpeakerTurn]
) -> list[TranscriptSegment]:
    diarization = list(turns)
    assigned: list[TranscriptSegment] = []
    for segment in segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in diarization:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker_id
        assigned.append(segment.model_copy(update={"speaker_id": best_speaker}))
    return assigned


def _warn_diarization_fallback(exc: Exception) -> None:
    message = (
        "Speaker diarization is unavailable; returning the transcript with "
        f"speaker_id='UNKNOWN'. Reason: {type(exc).__name__}: {exc}"
    )
    LOGGER.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
