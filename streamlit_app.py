from __future__ import annotations

import re
from datetime import date

import streamlit as st

from app.demo import demo_meeting_result
from app.models.schemas import MeetingResult
from app.services.docx_builder import build_docx
from app.services.pipeline import PipelineStageError, run_local_pipeline
from app.ui_state import apply_review_edits


st.set_page_config(
    page_title="QazMeeting AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #687086;
        --surface: rgba(255, 255, 255, 0.92);
        --line: #E4E7F0;
        --brand: #635BFF;
        --brand-dark: #4238D5;
        --teal: #1FA7A0;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(99, 91, 255, 0.11), transparent 25rem),
            radial-gradient(circle at 92% 8%, rgba(31, 167, 160, 0.09), transparent 23rem),
            #F6F7FB;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .qaz-hero {
        position: relative;
        overflow: hidden;
        padding: 2.15rem 2.35rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(128deg, #24234F 0%, #5149D8 52%, #168C91 130%);
        box-shadow: 0 20px 55px rgba(46, 43, 122, 0.22);
        margin-bottom: 1.35rem;
    }

    .qaz-hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -72px;
        top: -105px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.09);
    }

    .qaz-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .38rem .72rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }

    .qaz-hero h1 {
        margin: .9rem 0 .45rem;
        color: white;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.02;
        letter-spacing: -.045em;
    }

    .qaz-hero p {
        margin: 0;
        max-width: 690px;
        color: rgba(255,255,255,.82);
        font-size: 1.03rem;
        line-height: 1.55;
    }

    .qaz-feature-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.25rem;
    }

    .qaz-feature {
        padding: .43rem .68rem;
        border-radius: 10px;
        background: rgba(255,255,255,.11);
        color: rgba(255,255,255,.92);
        font-size: .82rem;
        font-weight: 600;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 18px;
        background: var(--surface);
        box-shadow: 0 8px 28px rgba(31, 38, 69, 0.055);
    }

    [data-testid="stMetric"] {
        min-height: 116px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 8px 24px rgba(31, 38, 69, 0.045);
    }

    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(120deg, var(--brand), var(--brand-dark));
        box-shadow: 0 9px 22px rgba(99, 91, 255, .24);
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 27px rgba(99, 91, 255, .31);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #AAA6F5;
        border-radius: 15px;
        background: #F7F6FF;
    }

    [data-baseweb="tab-list"] {
        gap: .35rem;
        padding: .35rem;
        border-radius: 14px;
        background: #EBEDF5;
    }

    [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        background: white;
        box-shadow: 0 3px 12px rgba(31, 38, 69, .09);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 15px;
    }

    .qaz-section-label {
        margin: 1.6rem 0 .65rem;
        color: #5149D8;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .qaz-export-copy {
        color: var(--muted);
        line-height: 1.5;
        margin-top: -.35rem;
    }

    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
        .qaz-hero { padding: 1.55rem 1.3rem; border-radius: 19px; }
        .qaz-feature-row { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" ._")
    return f"{(cleaned or 'meeting_protocol')[:80]}.docx"


def _store_result(result: MeetingResult, source_label: str, kind: str) -> None:
    for key in ("action_editor", "transcript_editor", "reviewed_result"):
        st.session_state.pop(key, None)
    st.session_state.result = result
    st.session_state.result_source = source_label
    st.session_state.result_kind = kind


def _render_result(result: MeetingResult, source_label: str) -> None:
    st.markdown('<div class="qaz-section-label">Результат обработки</div>', unsafe_allow_html=True)
    st.caption(source_label)

    review_count = sum(item.needs_review for item in result.action_items)
    named_speakers = {
        segment.speaker_name or segment.speaker_id
        for segment in result.transcript
        if segment.speaker_name or segment.speaker_id != "UNKNOWN"
    }
    metric_columns = st.columns(4)
    metric_columns[0].metric("Поручения", len(result.action_items))
    metric_columns[1].metric("Требуют проверки", review_count)
    metric_columns[2].metric("Реплики", len(result.transcript))
    metric_columns[3].metric("Говорящие", len(named_speakers))

    overview_tab, actions_tab, transcript_tab, events_tab = st.tabs(
        ["Обзор", "Поручения", "Транскрипт", "Журнал агента"]
    )

    with overview_tab:
        st.subheader("Саммари")
        st.write(result.summary or "Саммари не сформировано.")
        if result.warnings:
            st.markdown("#### Что требует внимания")
            for warning in result.warnings:
                st.warning(warning)
        else:
            st.success("Предупреждений нет. Результат готов к финальной проверке.")

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
    with actions_tab:
        st.caption("Исполнитель и срок доступны для ручного исправления перед экспортом.")
        if action_rows:
            edited_actions = st.data_editor(
                action_rows,
                key="action_editor",
                width="stretch",
                hide_index=True,
                disabled=["Суть", "Уверенность", "Статус проверки"],
                column_config={
                    "Исполнитель": st.column_config.TextColumn(
                        help="Можно исправить перед экспортом"
                    ),
                    "Срок": st.column_config.TextColumn(
                        help="Можно исправить перед экспортом"
                    ),
                    "Уверенность": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    ),
                },
            )
        else:
            st.info("Агент не выделил подтверждённых поручений.")
            edited_actions = []

    transcript_rows = [
        {
            "Таймкод": f"{_format_time(segment.start)}-{_format_time(segment.end)}",
            "Speaker ID": segment.speaker_id,
            "Имя говорящего": segment.speaker_name or "",
            "Текст": segment.text,
        }
        for segment in result.transcript
    ]
    with transcript_tab:
        st.caption("Имя говорящего можно уточнить; таймкоды и исходный текст защищены от изменений.")
        if transcript_rows:
            edited_transcript = st.data_editor(
                transcript_rows,
                key="transcript_editor",
                width="stretch",
                hide_index=True,
                disabled=["Таймкод", "Speaker ID", "Текст"],
                column_config={
                    "Имя говорящего": st.column_config.TextColumn(
                        help="Можно исправить перед экспортом"
                    )
                },
            )
        else:
            st.info("Транскрипт отсутствует.")
            edited_transcript = []

    reviewed = apply_review_edits(result, edited_actions, edited_transcript)
    st.session_state.reviewed_result = reviewed

    with events_tab:
        if not result.events:
            st.info("Агент не передал журнал событий.")
        for event in result.events:
            icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(event.status, "ℹ️")
            with st.container(border=True):
                st.markdown(f"{icon} **{event.stage}**")
                st.caption(event.message)

    docx_bytes = build_docx(reviewed)
    st.markdown('<div class="qaz-section-label">Экспорт протокола</div>', unsafe_allow_html=True)
    with st.container(border=True):
        export_copy, export_action = st.columns([2.4, 1])
        with export_copy:
            st.markdown("### Готово для Word")
            st.markdown(
                '<div class="qaz-export-copy">DOCX будет собран из текущих значений таблиц, '
                "включая все ручные исправления.</div>",
                unsafe_allow_html=True,
            )
        with export_action:
            st.download_button(
                "Скачать DOCX",
                data=docx_bytes,
                file_name=_safe_filename(result.title),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                width="stretch",
            )


st.markdown(
    """
    <section class="qaz-hero">
        <span class="qaz-eyebrow">● Local first · HackAlem AI</span>
        <h1>QazMeeting AI</h1>
        <p>Превращает локальную запись совещания в проверяемый протокол: саммари,
        поручения, сроки, говорящие и готовый DOCX.</p>
        <div class="qaz-feature-row">
            <span class="qaz-feature">MP3 и WAV</span>
            <span class="qaz-feature">Русский и қазақша</span>
            <span class="qaz-feature">Ручная проверка</span>
            <span class="qaz-feature">Локальная обработка</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="qaz-section-label">Новое совещание</div>', unsafe_allow_html=True)
with st.container(border=True):
    left, right = st.columns([2, 1])
    with left:
        title = st.text_input(
            "Название совещания",
            value="Рабочее совещание",
            placeholder="Например, Еженедельный статус проекта",
        )
    with right:
        meeting_date = st.date_input("Дата совещания", value=date.today())

    mode = st.radio(
        "Режим работы",
        ("Демо интерфейса", "Локальная обработка аудио"),
        horizontal=True,
        help="Демо использует синтетический MeetingResult и не обрабатывает загруженный файл.",
    )

if mode == "Демо интерфейса":
    st.info("Демо использует синтетические данные и никогда не выдаётся за результат аудио.")
    if st.button("Открыть демонстрационный протокол", type="primary"):
        with st.status("Проверка интерфейса", expanded=True) as status:
            st.write("Создание тестового результата")
            result = demo_meeting_result(title, meeting_date)
            st.write("Подготовка редактируемых таблиц и DOCX")
            _store_result(
                result,
                "Демо-данные — не результат обработки аудио",
                "demo",
            )
            status.update(label="Демо готово", state="complete", expanded=False)
else:
    upload = st.file_uploader(
        "Аудиозапись совещания",
        type=["mp3", "wav"],
        help="Поддерживаются MP3 и WAV до 500 МБ.",
    )
    st.caption("Файл обрабатывается локально во временной папке и удаляется после завершения.")
    if st.button(
        "Начать локальную обработку",
        type="primary",
        disabled=upload is None,
    ):
        for key in (
            "result",
            "result_source",
            "result_kind",
            "action_editor",
            "transcript_editor",
            "reviewed_result",
        ):
            st.session_state.pop(key, None)
        stage_status = st.status("Локальная обработка", expanded=True)
        try:
            result = run_local_pipeline(
                upload,
                upload.name,
                meeting_date,
                title,
                on_stage=stage_status.write,
            )
        except PipelineStageError as exc:
            stage_status.update(label=f"Ошибка: {exc.stage}", state="error", expanded=True)
            st.error(f"{exc.stage}: {exc.detail}")
        except Exception as exc:
            stage_status.update(label="Непредвиденная ошибка", state="error", expanded=True)
            st.error(f"Обработка: {exc}")
        else:
            _store_result(result, f"Локально обработан файл: {upload.name}", "local")
            label = (
                "Обработка завершена с предупреждениями"
                if result.warnings
                else "Обработка завершена"
            )
            stage_status.update(label=label, state="complete", expanded=False)

expected_kind = "demo" if mode == "Демо интерфейса" else "local"
if st.session_state.get("result_kind") == expected_kind:
    _render_result(st.session_state.result, st.session_state.result_source)
