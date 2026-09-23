from __future__ import annotations

import json

import pytest

from app.services import audio_cloud


def test_cloud_audio_requires_explicit_upload(tmp_path, monkeypatch):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"audio")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="allow_upload"):
        audio_cloud.process_audio_cloud(str(path))


def test_cloud_audio_requires_key_before_request(tmp_path, monkeypatch):
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"audio")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        audio_cloud.process_audio_cloud(str(path), allow_upload=True)


def test_cloud_audio_maps_diarized_segments_without_inventing_names(tmp_path, monkeypatch):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"fake audio bytes")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "segments": [
                        {"start": 0.2, "end": 1.5, "text": " Привет. ", "speaker": "A"},
                        {"start": 1.6, "end": 2.9, "text": "Сәлем.", "speaker": "B"},
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = request.headers
        seen["body"] = request.data
        return Response()

    monkeypatch.setattr(audio_cloud, "urlopen", fake_urlopen)
    segments = audio_cloud.process_audio_cloud(str(path), allow_upload=True)

    assert seen["url"] == audio_cloud.TRANSCRIPTION_URL
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert b"gpt-4o-transcribe-diarize" in seen["body"]
    assert b"diarized_json" in seen["body"]
    assert b"fake audio bytes" in seen["body"]
    assert [item.speaker_id for item in segments] == ["A", "B"]
    assert [item.text for item in segments] == ["Привет.", "Сәлем."]
    assert all(item.speaker_name is None and item.language is None for item in segments)


def test_cloud_audio_marks_missing_speaker_unknown():
    with pytest.warns(RuntimeWarning, match="UNKNOWN"):
        segments = audio_cloud._segments_from_diarized_response(
            {"segments": [{"start": 0, "end": 1, "text": "Реплика", "speaker": None}]}
        )
    assert segments[0].speaker_id == "UNKNOWN"
