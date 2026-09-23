import json
from datetime import date

from app.agents.meeting_agent import MeetingProtocolAgent, ModelResponseError
from app.models.schemas import TranscriptSegment


class FakeAgent(MeetingProtocolAgent):
    def __init__(self, payload):
        super().__init__(base_url="http://127.0.0.1:11434")
        self.payload = payload

    def _call_ollama(self, prompt: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def fixed_transcript():
    return [
        TranscriptSegment(id=1, start=0, end=4, speaker_id="S1", speaker_name="Руководитель", text="Ерлан, подготовь претензию поставщику к среде."),
        TranscriptSegment(id=2, start=4, end=8, speaker_id="S1", speaker_name="Руководитель", text="Салтанат Ерболовна, соберите предложения по логистике."),
        TranscriptSegment(id=3, start=8, end=13, speaker_id="S1", speaker_name="Руководитель", text="Ботагоз Нурлановна, найдите альтернативного поставщика за две недели."),
        TranscriptSegment(id=4, start=13, end=17, speaker_id="S1", speaker_name="Руководитель", text="Ерлан, повторю: подготовь претензию поставщику к среде."),
    ]


def fixed_payload():
    return {
        "speakers": [{"speaker_id": "S1", "proposed_name": "Руководитель", "confidence": 0.8}],
        "action_items": [
            {"task": "Подготовить претензию поставщику", "assignee": "Ерлан", "author_speaker_id": "S1", "deadline_text": "к среде", "evidence_segment_id": 1, "confidence": 0.96, "needs_review": False},
            {"task": "Собрать предложения по логистике", "assignee": "Салтанат Ерболовна", "author_speaker_id": "S1", "deadline_text": None, "evidence_segment_id": 2, "confidence": 0.94, "needs_review": False},
            {"task": "Найти альтернативного поставщика", "assignee": "Ботагоз Нурлановна", "author_speaker_id": "S1", "deadline_text": "за две недели", "evidence_segment_id": 3, "confidence": 0.97, "needs_review": False},
            {"task": "Подготовить претензию поставщику", "assignee": "Ерлан", "author_speaker_id": "S1", "deadline_text": "к среде", "evidence_segment_id": 4, "confidence": 0.90, "needs_review": False},
        ],
        "summary": "Обсудили претензию поставщику, логистические предложения и поиск альтернативного поставщика.",
    }


def test_fixed_transcript_deadlines_dedup_and_evidence():
    meeting_date = date(2026, 9, 21)
    transcript = fixed_transcript()
    result = FakeAgent(fixed_payload()).run(transcript, meeting_date, "Fixed test")

    assert len(result.action_items) == 3

    erlan = next(item for item in result.action_items if item.assignee == "Ерлан")
    assert erlan.deadline_text == "к среде"
    assert erlan.deadline_iso == date(2026, 9, 23)

    saltanat = next(item for item in result.action_items if item.assignee == "Салтанат Ерболовна")
    assert saltanat.deadline_text is None
    assert saltanat.deadline_iso is None

    botagoz = next(item for item in result.action_items if item.assignee == "Ботагоз Нурлановна")
    assert botagoz.deadline_text == "за две недели"
    assert botagoz.deadline_iso == date(2026, 10, 5)

    source_texts = {segment.text for segment in transcript}
    assert all(item.evidence in source_texts for item in result.action_items)
    assert all(item.evidence_start is not None and item.evidence_end is not None for item in result.action_items)


def test_ambiguous_deadline_keeps_text_without_invented_date():
    transcript = [
        TranscriptSegment(id=1, start=0, end=3, speaker_id="S1", text="Ерлан, отправь претензию до конца недели.")
    ]
    payload = {
        "speakers": [],
        "action_items": [
            {"task": "Отправить претензию", "assignee": "Ерлан", "author_speaker_id": "S1", "deadline_text": "до конца недели", "evidence_segment_id": 1, "confidence": 0.95, "needs_review": False}
        ],
        "summary": "Ерлану поручено отправить претензию.",
    }
    result = FakeAgent(payload).run(transcript, date(2026, 9, 21), "Ambiguous")
    item = result.action_items[0]
    assert item.deadline_text == "до конца недели"
    assert item.deadline_iso is None
    assert item.needs_review is True


def test_unavailable_model_preserves_transcript_and_returns_no_invented_actions():
    class BrokenAgent(MeetingProtocolAgent):
        def __init__(self):
            super().__init__(base_url="http://127.0.0.1:11434")

        def _call_ollama(self, prompt: str) -> str:
            raise ModelResponseError("offline")

    transcript = fixed_transcript()
    result = BrokenAgent().run(transcript, date(2026, 9, 21), "Offline")
    assert result.transcript == transcript
    assert result.action_items == []
    assert result.warnings
    assert any(event.status == "warning" for event in result.events)


def test_external_ollama_url_is_rejected():
    try:
        MeetingProtocolAgent(base_url="https://example.com")
    except ValueError as exc:
        assert "localhost" in str(exc)
    else:
        raise AssertionError("external endpoint was not rejected")
