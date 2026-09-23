"""Shared agent boundary; feature/agent owns the extraction implementation."""

from datetime import date
from app.models.schemas import MeetingResult, TranscriptSegment


class MeetingProtocolAgent:
    def run(
        self,
        transcript: list[TranscriptSegment],
        meeting_date: date | None = None,
        title: str = "Совещание",
    ) -> MeetingResult:
        raise NotImplementedError(
            "Agent extraction is not implemented on main yet. "
            "Integrate feature/agent and run the end-to-end smoke test."
        )
