from __future__ import annotations

from datetime import date

from app.models.schemas import ActionItem, AgentEvent, MeetingResult, SpeakerInfo, TranscriptSegment


def demo_meeting_result(title: str, meeting_date: date | None) -> MeetingResult:
    """Return explicit synthetic data for reviewing the UI before integration."""

    return MeetingResult(
        title=title or "Демонстрационное совещание",
        meeting_date=meeting_date,
        summary=(
            "Команда согласовала подготовку локального прототипа и отдельную проверку "
            "качества транскрипции смешанной русско-казахской речи."
        ),
        speakers=[
            SpeakerInfo(speaker_id="SPEAKER_00", proposed_name="Айгуль", confidence=0.91),
            SpeakerInfo(speaker_id="SPEAKER_01", proposed_name="Ерлан", confidence=0.72),
        ],
        transcript=[
            TranscriptSegment(
                id=1,
                start=0.0,
                end=7.4,
                speaker_id="SPEAKER_00",
                speaker_name="Айгуль",
                text="Ерлан, подготовьте локальную демонстрацию к пятнице.",
                language="ru",
            ),
            TranscriptSegment(
                id=2,
                start=7.5,
                end=14.8,
                speaker_id="SPEAKER_01",
                speaker_name="Ерлан",
                text="Жақсы, жұмаға дейін дайындаймын.",
                language="kk",
            ),
            TranscriptSegment(
                id=3,
                start=15.0,
                end=24.0,
                speaker_id="SPEAKER_00",
                speaker_name=None,
                text="И отдельно проверьте mixed speech на тестовой записи.",
                language="ru",
            ),
        ],
        action_items=[
            ActionItem(
                task="Подготовить локальную демонстрацию",
                assignee="Ерлан",
                deadline_text="к пятнице",
                evidence="Ерлан, подготовьте локальную демонстрацию к пятнице.",
                evidence_start=0.0,
                evidence_end=7.4,
                confidence=0.91,
                needs_review=False,
            ),
            ActionItem(
                task="Проверить качество смешанной русско-казахской речи",
                assignee=None,
                deadline_text=None,
                evidence="И отдельно проверьте mixed speech на тестовой записи.",
                evidence_start=15.0,
                evidence_end=24.0,
                confidence=0.63,
                needs_review=True,
                warning="Не определены исполнитель и срок.",
            ),
        ],
        warnings=["Демо-данные: второй исполнитель и срок требуют ручной проверки."],
        events=[
            AgentEvent(stage="demo", message="Загружен тестовый MeetingResult", status="success"),
            AgentEvent(stage="validation", message="Одно поручение требует проверки", status="warning"),
        ],
    )
