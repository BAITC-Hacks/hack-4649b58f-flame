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
    monkeypatch.setenv("AUDIO_STT_BACKEND", "mlx")
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
    monkeypatch.setenv("AUDIO_STT_BACKEND", "mlx")
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


def test_windows_backend_consumes_lazy_segments(tmp_path, monkeypatch):
    calls = {}

    class Model:
        def __init__(self, name, **kwargs):
            calls.update(model=name, **kwargs)

        def transcribe(self, path, **kwargs):
            calls.update(kwargs)
            return iter([types.SimpleNamespace(start=0.5, end=2.0, text=" Проверить договор. ")]), types.SimpleNamespace(language="ru", language_probability=0.99)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(audio.platform, "system", lambda: "Windows")
    monkeypatch.setenv("AUDIO_STT_BACKEND", "auto")
    for name in ("AUDIO_STT_MODEL", "AUDIO_STT_DEVICE", "AUDIO_STT_COMPUTE_TYPE", "AUDIO_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(audio, "_diarize", lambda _: [])
    path = tmp_path / "test.wav"
    path.write_bytes(b"synthetic backend input")
    with pytest.warns(RuntimeWarning):
        result = audio.process_audio(str(path))
    assert result[0].text == "Проверить договор."
    assert result[0].start == 0.5
    assert result[0].speaker_id == "UNKNOWN"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"
    assert calls["model"] == "small"
    assert calls["language"] is None


def test_unknown_backend_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_STT_BACKEND", "invalid")
    with pytest.raises(ValueError, match="AUDIO_STT_BACKEND"):
        audio._transcribe(tmp_path / "test.wav")
