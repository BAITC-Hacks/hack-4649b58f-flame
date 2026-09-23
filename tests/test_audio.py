from __future__ import annotations

import sys
import types

import pytest

from app.services import audio


def _fake_stt_result():
    return {
        "language": "ru",
        "language_probability": 0.95,
        "segments": [
            {"start": 0.2, "end": 1.8, "text": " Первый вопрос. "},
            {"start": 2.0, "end": 3.0, "text": "Второй вопрос."},
        ],
    }


def test_process_audio_keeps_transcript_when_diarization_fails(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"not decoded by the fake backend")
    fake_mlx = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: _fake_stt_result())
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setattr(audio, "_diarize", lambda _path: (_ for _ in ()).throw(RuntimeError("no model")))

    with pytest.warns(RuntimeWarning, match="no model"):
        segments = audio.process_audio(str(audio_path))

    assert [segment.text for segment in segments] == ["Первый вопрос.", "Второй вопрос."]
    assert [segment.speaker_id for segment in segments] == ["UNKNOWN", "UNKNOWN"]
    assert all(segment.language == "ru" for segment in segments)


def test_process_audio_assigns_largest_overlap(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake")
    fake_mlx = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: _fake_stt_result())
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setattr(
        audio,
        "_diarize",
        lambda _path: [
            audio._SpeakerTurn(0.0, 1.5, "SPEAKER_00"),
            audio._SpeakerTurn(1.5, 3.2, "SPEAKER_01"),
        ],
    )

    segments = audio.process_audio(str(audio_path))

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment.id for segment in segments] == [1, 2]


def test_process_audio_rejects_unsupported_format(tmp_path):
    audio_path = tmp_path / "meeting.ogg"
    audio_path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        audio.process_audio(str(audio_path))
