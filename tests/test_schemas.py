from datetime import date
import pytest
from pydantic import ValidationError
from app.config import local_model_url
from app.models.schemas import ActionItem, MeetingResult, TranscriptSegment


def test_shared_contract():
    segment = TranscriptSegment(id=1, start=1.2, end=2.3, text="Поручение")
    action = ActionItem(task="Подготовить отчёт", evidence=segment.text, confidence=0.8)
    result = MeetingResult(title="Demo", meeting_date=date(2026, 9, 23), transcript=[segment], action_items=[action])
    assert result.transcript[0].speaker_id == "UNKNOWN"
    assert result.action_items[0].deadline_iso is None
    assert result.events == []


def test_invalid_timestamp_and_confidence():
    with pytest.raises(ValidationError):
        TranscriptSegment(id=1, start=2, end=1, text="bad")
    with pytest.raises(ValidationError):
        ActionItem(task="x", evidence="x", confidence=1.2)


def test_external_model_url_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.com")
    with pytest.raises(ValueError):
        local_model_url()
