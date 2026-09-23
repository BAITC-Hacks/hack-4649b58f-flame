# QazMeeting AI

Прототип системы автопротоколирования совещаний для HackAlem AI (владелец кейса: Самрук-Қазына).

## Проблема и целевая аудитория

Протоколы совещаний часто фиксируются вручную. Из-за этого поручения, сроки и ответственные могут теряться или искажаться. Целевая аудитория MVP — секретари совещаний, руководители и кураторы поручений.

## Целевой сценарий MVP

1. Пользователь загружает аудиозапись совещания.
2. Локальный модуль speech-to-text строит транскрипт на русском, казахском или смешанной речи.
3. Модуль диаризации разделяет реплики по говорящим.
4. Meeting Protocol Agent анализирует транскрипт и извлекает поручения.
5. Для каждого поручения сохраняются ответственный, срок, evidence-фрагмент и признак необходимости ручной проверки.
6. Формируется краткое управленческое саммари.
7. Пользователь просматривает результат и экспортирует итоговый протокол в DOCX.

## Обязательные требования кейса

- распознавание речи участников;
- русский язык;
- казахский язык;
- смешанная русско-казахская речь;
- диаризация;
- поручения с ответственным и сроком;
- итоговое саммари;
- экспорт протокола в DOCX/PDF.

## Приватность

Кейс требует возможности работы в закрытом контуре: аудио и текст совещания не должны передаваться во внешние облачные API.

Поэтому финальный demo-пipeline проектируется local-first/self-hosted.

В репозитории предусмотрены переменные для OpenAI и NVIDIA API, потому что у команды есть соответствующие ключи, но они **не должны использоваться для передачи аудио или текста совещания в финальном сценарии кейса**. Секреты никогда не коммитятся в Git.

## Архитектура

```text
Audio MP3/WAV
      |
      v
Audio / STT pipeline
  - preprocessing
  - local transcription
  - timestamps
      |
      v
Diarization
  - speaker segments
  - merge with transcript
      |
      v
MeetingProtocolAgent
  - speaker resolution
  - task extraction
  - deadline normalization
  - evidence validation
  - summary
      |
      v
MeetingResult (Pydantic)
      |
      +--> Streamlit UI
      |
      +--> DOCX export
```

## Структура репозитория

```text
app/
├── agents/
│   └── meeting_agent.py
├── models/
│   └── schemas.py
├── services/
├── config.py
└── ...

tests/
├── test_schemas.py
└── ...

.env.example
.gitignore
requirements.txt
README.md
```

## Контракт данных

Основные модели находятся в `app/models/schemas.py`:

- `TranscriptSegment` — одна реплика с таймкодами и speaker id;
- `ActionItem` — поручение с ответственным, сроком и evidence;
- `SpeakerInfo` — соответствие speaker id и предполагаемого имени;
- `MeetingResult` — итог анализа одного совещания;
- `AgentEvent` — журнал этапов работы агента.

Это общий контракт между тремя ветками разработки.

## Meeting Protocol Agent

Оркестратор находится в `app/agents/meeting_agent.py`.

Он построен вокруг независимых инструментов:

```text
Transcript
   |
   +--> speaker resolver
   +--> task extractor
   +--> deadline normalizer
   +--> validator
   +--> summarizer
   |
   v
MeetingResult
```

Архитектура специально допускает замену конкретной локальной модели без изменения UI и STT-pipeline.

## Разделение команды

- `feature/audio` — Audio / STT / diarization;
- `feature/agent` — Meeting Protocol Agent / extraction / validation / summary;
- `feature/ui` — Streamlit / DOCX / integration;
- `main` — интегрированная проверенная версия.

## Установка базового окружения

Рекомендуется Python 3.11 или 3.12.

```bash
git clone https://github.com/BAITC-Hacks/hack-4649b58f-flame.git
cd hack-4649b58f-flame

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows

pip install -r requirements.txt
cp .env.example .env
```

## Секреты

Файл `.env` находится в `.gitignore`.

Не коммитьте:

- `OPENAI_API_KEY`;
- `NVIDIA_API_KEY`;
- `HF_TOKEN`;
- любые другие токены;
- реальные чувствительные аудиозаписи.

## Текущий статус

На `main` создан архитектурный фундамент: общие Pydantic-схемы, конфигурация, интерфейсы Meeting Protocol Agent, базовые зависимости и правила хранения секретов.

STT, diarization, локальная LLM-реализация агента, Streamlit UI и DOCX export разрабатываются параллельно в feature-ветках и считаются реализованными только после фактического smoke-test и merge в `main`.

## Проверка фундамента

```bash
pytest -q
```

Тесты фундамента проверяют создание общего `MeetingResult` и базовые ограничения моделей.

## Данные и внешние сервисы

Для демонстрации используются предоставленные организаторами тестовые записи совещаний и отдельный короткий тест смешанной русско-казахской речи.

Внешние AI API не являются частью допустимого финального processing path для аудио/текста совещания из-за требования закрытого контура.

## Ограничения текущего состояния

- production-интеграция с Teams/Zoom/Google Meet не реализована;
- интеграция с СЭД не входит в обязательный MVP;
- voice biometric identification не входит в обязательный MVP;
- качество STT/diarization/agent extraction будет зафиксировано в README после реального end-to-end тестирования.

## Deployed version

Пока отсутствует. Для хакатона приоритет — воспроизводимый локальный demo.
