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
        --ink: #F4EEE8;
        --muted: #B8ADA4;
        --surface: rgba(35, 31, 28, 0.84);
        --line: rgba(255, 235, 220, 0.12);
        --brand: #D97745;
        --brand-dark: #A84F2E;
        --teal: #E6A15B;
    }

    @keyframes qaz-rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes qaz-float {
        0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-12px, 11px, 0) scale(1.035); }
    }

    @keyframes qaz-pulse {
        0%, 100% { opacity: .55; transform: scale(.82); }
        50% { opacity: 1; transform: scale(1.18); }
    }

    @keyframes qaz-wave {
        0%, 100% { height: 8px; opacity: .58; }
        50% { height: 28px; opacity: 1; }
    }

    @keyframes qaz-sheen {
        0% { transform: translateX(-140%) skewX(-18deg); }
        70%, 100% { transform: translateX(420%) skewX(-18deg); }
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 4%, rgba(217, 119, 69, 0.16), transparent 28rem),
            radial-gradient(circle at 92% 7%, rgba(230, 161, 91, 0.08), transparent 25rem),
            linear-gradient(180deg, #151311 0%, #191512 46%, #12100F 100%);
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
        background: linear-gradient(128deg, #191512 0%, #35231C 55%, #713923 130%);
        box-shadow: 0 24px 65px rgba(0, 0, 0, 0.34);
        margin-bottom: 1.35rem;
        animation: qaz-rise .9s cubic-bezier(.22,1,.36,1) both;
    }

    .qaz-hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -72px;
        top: -105px;
        border-radius: 50%;
        background: rgba(255, 214, 183, 0.07);
        animation: qaz-float 14s ease-in-out infinite;
        pointer-events: none;
    }

    .qaz-hero::before {
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        left: 56%;
        bottom: -132px;
        border: 1px solid rgba(255,220,194,.10);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 68px rgba(255,255,255,.025);
        pointer-events: none;
    }

    .qaz-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .38rem .72rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        background: rgba(255,235,220,.08);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }

    .qaz-eyebrow::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #F0A66C;
        box-shadow: 0 0 0 5px rgba(240,166,108,.12);
        animation: qaz-pulse 3.4s ease-in-out infinite;
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
        background: rgba(255,235,220,.075);
        color: rgba(255,255,255,.92);
        font-size: .82rem;
        font-weight: 600;
        transition: transform .38s cubic-bezier(.22,1,.36,1), background .38s ease;
    }

    .qaz-feature:hover {
        transform: translateY(-2px);
        background: rgba(255,235,220,.13);
    }

    .qaz-signal {
        position: absolute;
        z-index: 1;
        right: 2.4rem;
        bottom: 2.15rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        width: 86px;
        height: 44px;
        padding: 0 12px;
        border: 1px solid rgba(255,224,202,.14);
        border-radius: 14px;
        background: rgba(18,13,10,.28);
        backdrop-filter: blur(8px);
    }

    .qaz-signal span {
        width: 4px;
        min-height: 8px;
        border-radius: 99px;
        background: linear-gradient(180deg, #FFE7D2, #E38A52);
        animation: qaz-wave 1.8s ease-in-out infinite;
        animation-delay: calc(var(--i) * -0.19s);
    }

    .qaz-flow {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .7rem;
        margin: 0 0 1.35rem;
        animation: qaz-rise .95s .12s cubic-bezier(.22,1,.36,1) both;
    }

    .qaz-flow-item {
        display: flex;
        align-items: center;
        gap: .72rem;
        padding: .78rem .9rem;
        border: 1px solid rgba(255,235,220,.10);
        border-radius: 14px;
        background: rgba(35,31,28,.68);
        color: var(--muted);
        font-size: .82rem;
        font-weight: 600;
        backdrop-filter: blur(9px);
        transition: transform .4s cubic-bezier(.22,1,.36,1), border-color .4s ease, box-shadow .4s ease;
    }

    .qaz-flow-item:hover {
        transform: translateY(-3px);
        border-color: rgba(217,119,69,.34);
        box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }

    .qaz-flow-number {
        display: grid;
        place-items: center;
        flex: 0 0 27px;
        width: 27px;
        height: 27px;
        border-radius: 9px;
        background: linear-gradient(135deg, #D97745, #E6A15B);
        color: white;
        font-size: .72rem;
        box-shadow: 0 6px 16px rgba(168,79,46,.27);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 18px;
        background: var(--surface);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.16);
        backdrop-filter: blur(18px) saturate(115%);
        animation: qaz-rise .82s cubic-bezier(.22,1,.36,1) both;
        transition: border-color .36s ease, box-shadow .36s ease;
    }

    [data-testid="stMetric"] {
        min-height: 116px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(16px) saturate(115%);
        transition: transform .42s cubic-bezier(.22,1,.36,1), border-color .42s ease, box-shadow .42s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(217,119,69,.3);
        box-shadow: 0 18px 38px rgba(0,0,0,.24);
    }

    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(120deg, var(--brand), var(--brand-dark));
        box-shadow: 0 10px 26px rgba(168, 79, 46, .29);
        font-weight: 700;
        transition: transform .34s cubic-bezier(.22,1,.36,1), box-shadow .34s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button[kind="primary"]::after,
    .stDownloadButton > button[kind="primary"]::after {
        content: "";
        position: absolute;
        inset: -35% auto -35% -25%;
        width: 22%;
        background: rgba(255,255,255,.28);
        filter: blur(1px);
        animation: qaz-sheen 7.5s ease-in-out infinite;
        pointer-events: none;
    }

    .stButton > button[kind="primary"]:disabled::after {
        display: none;
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 15px 32px rgba(168, 79, 46, .38);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #9E624A;
        border-radius: 15px;
        background: #211B18;
    }

    [data-baseweb="tab-list"] {
        gap: .35rem;
        padding: .35rem;
        border-radius: 14px;
        background: #211D1A;
    }

    [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 1rem;
        padding-right: 1rem;
        transition: color .34s ease, background .34s ease, transform .34s cubic-bezier(.22,1,.36,1);
    }

    [data-baseweb="tab"]:hover {
        transform: translateY(-1px);
    }

    [aria-selected="true"][data-baseweb="tab"] {
        background: #352A24;
        box-shadow: 0 5px 16px rgba(0,0,0,.23);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 15px;
    }

    .qaz-section-label {
        margin: 1.6rem 0 .65rem;
        color: #E59A6A;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: .48rem;
        animation: qaz-rise .78s cubic-bezier(.22,1,.36,1) both;
    }

    .qaz-section-label::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--brand), var(--teal));
        box-shadow: 0 0 0 5px rgba(217,119,69,.10);
    }

    .qaz-export-copy {
        color: var(--muted);
        line-height: 1.5;
        margin-top: -.35rem;
    }

    @media (max-width: 900px) {
        .qaz-signal { display: none; }
    }

    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
        .qaz-hero { padding: 1.55rem 1.3rem; border-radius: 19px; }
        .qaz-feature-row { display: none; }
        .qaz-flow { grid-template-columns: 1fr; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
        }
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
        <span class="qaz-eyebrow">Local first · HackAlem AI</span>
        <h1>QazMeeting AI</h1>
        <p>Превращает локальную запись совещания в проверяемый протокол: саммари,
        поручения, сроки, говорящие и готовый DOCX.</p>
        <div class="qaz-feature-row">
            <span class="qaz-feature">MP3 и WAV</span>
            <span class="qaz-feature">Русский и қазақша</span>
            <span class="qaz-feature">Ручная проверка</span>
            <span class="qaz-feature">Локальная обработка</span>
        </div>
        <div class="qaz-signal" aria-hidden="true">
            <span style="--i:1"></span><span style="--i:2"></span>
            <span style="--i:3"></span><span style="--i:4"></span>
            <span style="--i:5"></span><span style="--i:6"></span>
        </div>
    </section>
    <div class="qaz-flow" aria-label="Этапы подготовки протокола">
        <div class="qaz-flow-item"><span class="qaz-flow-number">01</span>Распознавание речи</div>
        <div class="qaz-flow-item"><span class="qaz-flow-number">02</span>Локальный анализ</div>
        <div class="qaz-flow-item"><span class="qaz-flow-number">03</span>Проверка и DOCX</div>
    </div>
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
