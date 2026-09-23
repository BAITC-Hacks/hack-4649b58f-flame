from io import BytesIO

import pytest

from app.demo import demo_meeting_result
from app.services.audio import process_audio
from app.services.pipeline import PipelineStageError, resolve_audio_processor, run_local_pipeline


class FakeAgent:
    def run(self, segments, meeting_date, title):
        result = demo_meeting_result(title, meeting_date)
        return result.model_copy(update={"transcript": segments})


def test_audio_processor_uses_confirmed_team_import():
    assert resolve_audio_processor() is process_audio


def test_local_pipeline_uses_audio_segments_and_agent():
    expected = demo_meeting_result("x", None).transcript

    result = run_local_pipeline(
        BytesIO(b"RIFF-test"),
        "meeting.wav",
        None,
        "Реальное совещание",
        audio_processor=lambda _: expected,
        agent=FakeAgent(),
    )

    assert result.title == "Реальное совещание"
    assert result.transcript == expected


def test_local_pipeline_reports_audio_stage():
    def fail(_):
        raise RuntimeError("модель недоступна")

    with pytest.raises(PipelineStageError, match="Распознавание аудио"):
        run_local_pipeline(
            BytesIO(b"ID3-test"),
            "meeting.mp3",
            None,
            "Совещание",
            audio_processor=fail,
            agent=FakeAgent(),
        )
