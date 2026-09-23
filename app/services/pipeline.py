from __future__ import annotations

import importlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO

from app.agents.meeting_agent import MeetingProtocolAgent
from app.models.schemas import MeetingResult, TranscriptSegment


AudioProcessor = Callable[[str], list[TranscriptSegment]]


@dataclass(slots=True)
class PipelineStageError(RuntimeError):
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"{self.stage}: {self.detail}"


def resolve_audio_processor() -> AudioProcessor:
    """Find the local audio team's public function without copying its implementation."""

    candidates = (
        ("app.services.audio", "process_audio"),
        ("app.services.audio_pipeline", "process_audio"),
        ("app.audio", "process_audio"),
    )
    errors: list[str] = []
    for module_name, attribute in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                errors.append(f"{module_name}: отсутствует зависимость {exc.name}")
            continue
        function = getattr(module, attribute, None)
        if callable(function):
            return function
        errors.append(f"{module_name}: нет функции {attribute}")

    suffix = f" Детали: {'; '.join(errors)}" if errors else ""
    raise PipelineStageError(
        "Распознавание аудио",
        "process_audio пока не подключён. Ожидается локальная функция в ветке feature/audio."
        + suffix,
    )


def run_local_pipeline(
    uploaded_file: BinaryIO,
    filename: str,
    meeting_date: date | None,
    title: str,
    *,
    audio_processor: AudioProcessor | None = None,
    agent: MeetingProtocolAgent | None = None,
) -> MeetingResult:
    """Run the local-only audio and agent pipeline for one uploaded file."""

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

        processor = audio_processor or resolve_audio_processor()
        try:
            segments = processor(str(temp_path))
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError("Распознавание аудио", str(exc)) from exc

        if not segments:
            raise PipelineStageError("Распознавание аудио", "Модуль не вернул ни одного сегмента.")

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
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

