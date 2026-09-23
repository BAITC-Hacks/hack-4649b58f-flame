from __future__ import annotations

from datetime import date

import streamlit as st

from app.demo import demo_meeting_result
from app.models.schemas import MeetingResult
from app.services.docx_builder import build_docx
from app.services.pipeline import PipelineStageError, run_local_pipeline
from app.ui_state import apply_review_edits


st.set_page_config(page_title="QazMeeting AI", page_icon="📝", layout="wide")


def _format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _render_result(result: MeetingResult, source_label: str) -> None:
    st.caption(source_label)
    st.subheader("Саммари")
    st.write(result.summary or "Саммари не сформировано.")

    if result.warnings:
        with st.expander(f"Предупреждения ({len(result.warnings)})", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    st.subheader("Поручения")
    action_rows = [
        {
            "Суть": item.task,
            "Исполнитель": item.assignee or "",
            "Срок": item.deadline_text or (item.deadline_iso.isoformat() if item.deadline_iso else ""),
            "Уверенность": round(item.confidence * 100),
            "Статус проверки": "Требует проверки" if item.needs_review else "Проверено",
        }
        for item in result.action_items
    ]
    edited_actions = st.data_editor(
        action_rows,
        key="action_editor",
        width="stretch",
        hide_index=True,
        disabled=["Суть", "Уверенность", "Статус проверки"],
        column_config={
            "Исполнитель": st.column_config.TextColumn(help="Можно исправить перед экспортом"),
            "Срок": st.column_config.TextColumn(help="Можно исправить перед экспортом"),
            "Уверенность": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
        },
    )

    st.subheader("Транскрипт")
    transcript_rows = [
        {
            "Таймкод": f"{_format_time(segment.start)}-{_format_time(segment.end)}",
            "Speaker ID": segment.speaker_id,
            "Имя говорящего": segment.speaker_name or "",
            "Текст": segment.text,
        }
        for segment in result.transcript
    ]
    edited_transcript = st.data_editor(
        transcript_rows,
        key="transcript_editor",
        width="stretch",
        hide_index=True,
        disabled=["Таймкод", "Speaker ID", "Текст"],
        column_config={
            "Имя говорящего": st.column_config.TextColumn(help="Можно исправить перед экспортом")
        },
    )

    reviewed = apply_review_edits(result, edited_actions, edited_transcript)
    st.session_state.reviewed_result = reviewed

    with st.expander(f"Журнал событий агента ({len(result.events)})"):
        if not result.events:
            st.caption("Агент не передал журнал событий.")
        for event in result.events:
            icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(event.status, "ℹ️")
            st.write(f"{icon} **{event.stage}** — {event.message}")

    docx_bytes = build_docx(reviewed)
    st.download_button(
        "Скачать исправленный DOCX",
        data=docx_bytes,
        file_name="meeting_protocol.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
    )


st.title("QazMeeting AI")
st.write("Локальная подготовка протокола совещания с ручной проверкой перед экспортом.")

left, right = st.columns([2, 1])
with left:
    title = st.text_input("Название совещания", value="Рабочее совещание")
with right:
    meeting_date = st.date_input("Дата совещания", value=date.today())

mode = st.radio(
    "Режим",
    ("Демо интерфейса", "Локальная обработка аудио"),
    horizontal=True,
    help="Демо использует синтетический MeetingResult и не обрабатывает загруженный файл.",
)

if mode == "Демо интерфейса":
    st.info("Демо-режим: показаны только синтетические данные. MP3/WAV не обрабатывается.")
    if st.button("Загрузить тестовый MeetingResult", type="primary"):
        with st.status("Проверка интерфейса", expanded=True) as status:
            st.write("Создание тестового результата")
            result = demo_meeting_result(title, meeting_date)
            st.write("Подготовка редактируемых таблиц и DOCX")
            st.session_state.result = result
            st.session_state.result_source = "Демо-данные — не результат обработки аудио"
            status.update(label="Демо готово", state="complete", expanded=False)
else:
    upload = st.file_uploader("Загрузите аудио", type=["mp3", "wav"])
    st.caption("Файл сохраняется только во временную локальную папку и удаляется после обработки.")
    if st.button("Обработать аудио", type="primary", disabled=upload is None):
        stage_status = st.status("Локальная обработка", expanded=True)
        try:
            stage_status.write("1/3 Сохранение файла во временную локальную папку")
            stage_status.write("2/3 Распознавание речи и диаризация")
            stage_status.write("3/3 Анализ транскрипта локальным агентом")
            result = run_local_pipeline(upload, upload.name, meeting_date, title)
        except PipelineStageError as exc:
            stage_status.update(label=f"Ошибка: {exc.stage}", state="error", expanded=True)
            st.error(f"{exc.stage}: {exc.detail}")
        except Exception as exc:
            stage_status.update(label="Непредвиденная ошибка", state="error", expanded=True)
            st.error(f"Обработка: {exc}")
        else:
            st.session_state.result = result
            st.session_state.result_source = f"Локально обработан файл: {upload.name}"
            stage_status.update(label="Обработка завершена", state="complete", expanded=False)

if "result" in st.session_state:
    _render_result(st.session_state.result, st.session_state.result_source)

