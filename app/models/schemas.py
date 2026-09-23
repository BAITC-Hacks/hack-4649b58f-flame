from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker_id: str = "UNKNOWN"
    speaker_name: str | None = None
    text: str
    language: str | None = None

    @model_validator(mode="after")
    def check_interval(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class ActionItem(BaseModel):
    task: str
    assignee: str | None = None
    deadline_text: str | None = None
    deadline_iso: date | None = None
    evidence: str
    evidence_start: float | None = None
    evidence_end: float | None = None
    confidence: float = Field(ge=0, le=1)
    needs_review: bool = False
    warning: str | None = None


class SpeakerInfo(BaseModel):
    speaker_id: str
    proposed_name: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)


class AgentEvent(BaseModel):
    stage: str
    message: str
    status: str = "info"


class MeetingResult(BaseModel):
    title: str
    meeting_date: date | None = None
    summary: str = ""
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
