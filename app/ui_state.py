from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.models.schemas import ActionItem, MeetingResult, TranscriptSegment


def apply_review_edits(
    result: MeetingResult,
    action_rows: Sequence[Mapping[str, object]],
    transcript_rows: Sequence[Mapping[str, object]],
) -> MeetingResult:
    """Apply reviewer-editable UI fields while preserving the shared schema."""

    action_items: list[ActionItem] = []
    for original, row in zip(result.action_items, action_rows):
        assignee = str(row.get("Исполнитель") or "").strip() or None
        deadline = str(row.get("Срок") or "").strip() or None
        action_items.append(
            original.model_copy(
                update={
                    "assignee": assignee,
                    "deadline_text": deadline,
                    "deadline_iso": None,
                }
            )
        )
    action_items.extend(result.action_items[len(action_items) :])

    transcript: list[TranscriptSegment] = []
    for original, row in zip(result.transcript, transcript_rows):
        speaker_name = str(row.get("Имя говорящего") or "").strip() or None
        transcript.append(original.model_copy(update={"speaker_name": speaker_name}))
    transcript.extend(result.transcript[len(transcript) :])

    return result.model_copy(update={"action_items": action_items, "transcript": transcript})
