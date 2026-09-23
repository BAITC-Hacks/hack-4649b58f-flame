from __future__ import annotations

import tempfile
import warnings as py_warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO

from app.agents.meeting_agent import MeetingProtocolAgent
from app.models.schemas import AgentEvent, MeetingResult, TranscriptSegment


AudioProcessor = Callable[[str], list[TranscriptSegment]]
StageReporter = Callable[[str], None]


@dataclass(slots=True)
class PipelineStageError(RuntimeError):
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"{self.stage}: {self.detail}"


def resolve_audio_processor() -> AudioProcessor:
    """Import the audio team's confirmed public function without copying its logic."""

    try:
        from app.services.audio import process_audio
    except (ImportError, ModuleNotFoundError) as exc:
        raise PipelineStageError(
            "Распознавание аудио",
            f"Не удалось импортировать app.services.audio.process_audio: {exc}",
        ) from exc
    return process_audio


def run_local_pipeline(
    uploaded_file: BinaryIO,
    filename: str,
    meeting_date: date | None,
    title: str,
    *,
    audio_processor: AudioProcessor | None = None,
    agent: MeetingProtocolAgent | None = None,
    on_stage: StageReporter | None = None,
) -> MeetingResult:
    """Run the local-only audio and agent pipeline for one uploaded file."""

    report = on_stage or (lambda _message: None)

    report("1/3 Проверка и локальное сохранение файла")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        raise PipelineStageError("Загрузка файла", "Поддерживаются только MP3 и WAV.")

    payload = uploaded_file.read()
    if not payload:
        raise PipelineStageError("Загрузка файла", "Загруженный файл пуст.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="qazmeeting_", suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)

        report("2/3 Распознавание речи и диаризация")
        processor = audio_processor or resolve_audio_processor()
        try:
            with py_warnings.catch_warnings(record=True) as caught_warnings:
                py_warnings.simplefilter("always")
                segments = processor(str(temp_path))
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError("Распознавание аудио", str(exc)) from exc

        if not segments:
            raise PipelineStageError("Распознавание аудио", "Модуль не вернул ни одного сегмента.")

        audio_warnings = [str(item.message) for item in caught_warnings]
        if segments and all(segment.speaker_id == "UNKNOWN" for segment in segments):
            fallback = "Диаризация недоступна: транскрипт сохранён без определения говорящих."
            if not any("diar" in message.casefold() for message in audio_warnings):
                audio_warnings.append(fallback)

        report("3/3 Анализ транскрипта локальным агентом")
        protocol_agent = agent or MeetingProtocolAgent()
        try:
            result = protocol_agent.run(segments, meeting_date, title)
        except Exception as exc:
            raise PipelineStageError("Анализ транскрипта", str(exc)) from exc

        if not isinstance(result, MeetingResult):
            raise PipelineStageError(
                "Анализ транскрипта",
                "Агент вернул результат, не соответствующий MeetingResult.",
            )
        if audio_warnings:
            result = result.model_copy(
                update={
                    "warnings": [*audio_warnings, *result.warnings],
                    "events": [
                        AgentEvent(
                            stage="diarization",
                            message=message,
                            status="warning",
                        )
                        for message in audio_warnings
                    ]
                    + result.events,
                }
            )
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
