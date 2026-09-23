"""Meeting protocol extraction using only a local Ollama endpoint."""
from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from app.config import local_model_url
from app.models.schemas import ActionItem, AgentEvent, MeetingResult, SpeakerInfo, TranscriptSegment

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
WORDS = re.compile(r"[0-9A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+")
WEEKDAYS = {"понедельнику": 0, "вторнику": 1, "среде": 2, "четвергу": 3, "пятнице": 4, "субботе": 5, "воскресенью": 6}
NUMBERS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}


class ModelResponseError(RuntimeError):
    pass


class MeetingProtocolAgent:
    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 120) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or local_model_url()).rstrip("/")
        self.timeout = timeout
        host = urlparse(self.base_url).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama endpoint must point to localhost")

    def run(self, transcript: list[TranscriptSegment], meeting_date: date | None = None, title: str = "Совещание") -> MeetingResult:
        events = [AgentEvent(stage="input", message=f"Получено реплик: {len(transcript)}; исходный транскрипт сохранён.", status="success")]
        warnings: list[str] = []
        source = list(transcript)
        if not source:
            warnings.append("Транскрипт пуст: поручения не извлекались.")
            events.append(AgentEvent(stage="validation", message=warnings[-1], status="warning"))
            return MeetingResult(title=title, meeting_date=meeting_date, transcript=source, warnings=warnings, events=events)

        try:
            events.append(AgentEvent(stage="local_model", message=f"Локальная модель: {self.model}."))
            raw = self._call_ollama(self._prompt(source, meeting_date, title))
            payload = self._parse(raw)
            events.append(AgentEvent(stage="parse", message="JSON локальной модели разобран.", status="success"))
        except Exception as exc:
            msg = f"Локальная модель недоступна или вернула непроверяемый ответ: {type(exc).__name__}: {exc}"
            warnings.append(msg)
            events.append(AgentEvent(stage="local_model", message=msg, status="warning"))
            return MeetingResult(title=title, meeting_date=meeting_date, transcript=source, warnings=warnings, events=events)

        speakers = self._speakers(payload.get("speakers", []), source, warnings)
        actions = self._actions(payload.get("action_items", []), source, meeting_date, warnings)
        before = len(actions)
        actions = self._dedupe(actions)
        events.extend([
            AgentEvent(stage="speakers", message=f"Говорящих: {len(speakers)}.", status="success"),
            AgentEvent(stage="validation", message=f"Evidence подтверждено: {before}; дублей удалено: {before-len(actions)}.", status="success"),
            AgentEvent(stage="deadlines", message="Точные даты выставлены только для однозначных сроков.", status="success"),
            AgentEvent(stage="summary", message="Саммари сформировано.", status="success"),
        ])
        if warnings:
            events.append(AgentEvent(stage="warnings", message=f"Предупреждений: {len(warnings)}.", status="warning"))
        return MeetingResult(title=title, meeting_date=meeting_date, summary=self._summary(payload.get("summary"), actions), speakers=speakers, transcript=source, action_items=actions, warnings=warnings, events=events)

    def _call_ollama(self, prompt: str) -> str:
        data = json.dumps({
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 16384},
            "messages": [
                {"role": "system", "content": "Извлекай только подтверждённые транскриптом факты. Не выдумывай имена, поручения или сроки. Верни только JSON."},
                {"role": "user", "content": prompt},
            ],
        }, ensure_ascii=False).encode()
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                envelope = json.loads(response.read().decode())
            content = envelope["message"]["content"]
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelResponseError(f"Ollama {self.base_url} недоступен или вернул неверный формат") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError("пустой ответ Ollama")
        return content

    def _prompt(self, transcript: list[TranscriptSegment], meeting_date: date | None, title: str) -> str:
        rows = [s.model_dump() for s in transcript]
        schema = {"speakers": [{"speaker_id": "S1", "proposed_name": None, "confidence": 0.0}], "action_items": [{"task": "", "assignee": None, "author_speaker_id": "S1", "deadline_text": None, "evidence_segment_id": 1, "confidence": 0.0, "needs_review": False}], "summary": ""}
        return f"""Название: {title}\nДата: {meeting_date.isoformat() if meeting_date else 'не указана'}\n
Верни JSON по схеме: {json.dumps(schema, ensure_ascii=False)}
Правила:
- поручение только если есть явное задание/просьба/обязательство с действием;
- author_speaker_id — автор реплики-поручения, assignee — исполнитель; не путай их;
- если исполнитель или срок не названы, ставь null;
- deadline_text копируй дословно из evidence, дату не вычисляй;
- evidence_segment_id должен указывать на реплику, подтверждающую поручение;
- повторённое поручение верни один раз;
- имя говорящего предлагай только при текстовом основании, иначе null; сомнение отражай confidence;
- needs_review=true при двусмысленности;
- summary: 1-3 коротких предложения только по транскрипту.
Транскрипт: {json.dumps(rows, ensure_ascii=False)}"""

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        text = text.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text[len(fence):].lstrip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            if text.endswith(fence):
                text = text[:-len(fence)].rstrip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseError("ответ не JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("speakers", []), list) or not isinstance(payload.get("action_items", []), list):
            raise ModelResponseError("неверная структура JSON")
        return payload

    def _speakers(self, raw: list[Any], transcript: list[TranscriptSegment], warnings: list[str]) -> list[SpeakerInfo]:
        ids = {s.speaker_id for s in transcript}
        all_text = " ".join(s.text for s in transcript)
        supplied = {s.speaker_name for s in transcript if s.speaker_name}
        out: dict[str, SpeakerInfo] = {}
        for x in raw:
            if not isinstance(x, dict) or str(x.get("speaker_id", "")) not in ids:
                continue
            sid = str(x["speaker_id"])
            name = self._clean(x.get("proposed_name"))
            conf = self._conf(x.get("confidence"), 0)
            if name and name not in supplied and not self._contains(name, all_text):
                warnings.append(f"Имя {name!r} для {sid} не подтверждено текстом; оставлено null.")
                name, conf = None, 0
            out[sid] = SpeakerInfo(speaker_id=sid, proposed_name=name, confidence=conf)
        for sid in ids:
            if sid not in out:
                names = {s.speaker_name for s in transcript if s.speaker_id == sid and s.speaker_name}
                out[sid] = SpeakerInfo(speaker_id=sid, proposed_name=next(iter(names)) if len(names) == 1 else None, confidence=0.8 if len(names) == 1 else 0)
        return list(out.values())

    def _actions(self, raw: list[Any], transcript: list[TranscriptSegment], meeting_date: date | None, warnings: list[str]) -> list[ActionItem]:
        by_id = {s.id: s for s in transcript}
        out: list[ActionItem] = []
        for n, x in enumerate(raw, 1):
            if not isinstance(x, dict):
                continue
            task, eid = self._clean(x.get("task")), x.get("evidence_segment_id")
            if not task or not isinstance(eid, int) or eid not in by_id:
                warnings.append(f"Поручение #{n} пропущено: нет валидного task/evidence.")
                continue
            ev = by_id[eid]
            if not self._grounded(task, ev.text):
                warnings.append(f"Поручение #{n} пропущено: задача не подтверждается evidence.")
                continue
            review = bool(x.get("needs_review", False))
            notes: list[str] = []
            author = self._clean(x.get("author_speaker_id"))
            if author and author != ev.speaker_id:
                review = True; notes.append("автор не совпадает со speaker evidence")
            assignee = self._clean(x.get("assignee"))
            if assignee and not self._contains(assignee, ev.text) and not self._contains(self._norm(assignee).split()[0], ev.text):
                assignee = None; review = True; notes.append("исполнитель не подтверждён evidence")
            deadline_text = self._clean(x.get("deadline_text"))
            if deadline_text and not self._contains(deadline_text, ev.text):
                deadline_text = None; review = True; notes.append("текст срока не найден в evidence")
            deadline_iso, ambiguous = self._deadline(deadline_text, meeting_date)
            if ambiguous:
                review = True; notes.append("срок неоднозначен: точная дата не ставилась")
            confidence = self._conf(x.get("confidence"), 0.5)
            if confidence < 0.7:
                review = True; notes.append("низкая уверенность")
            warning = "; ".join(notes) or None
            if warning:
                warnings.append(f"{task}: {warning}.")
            out.append(ActionItem(task=task, assignee=assignee, deadline_text=deadline_text, deadline_iso=deadline_iso, evidence=ev.text, evidence_start=ev.start, evidence_end=ev.end, confidence=confidence, needs_review=review, warning=warning))
        return out

    def _deadline(self, raw: str | None, meeting_date: date | None) -> tuple[date | None, bool]:
        if not raw: return None, False
        if not meeting_date: return None, True
        t = self._norm(raw)
        m = re.search(r"\b(20\d{2}) (\d{1,2}) (\d{1,2})\b", t)
        if m:
            try: return date(*map(int, m.groups())), False
            except ValueError: return None, True
        if "послезавтра" in t: return meeting_date + timedelta(days=2), False
        if "завтра" in t: return meeting_date + timedelta(days=1), False
        if "сегодня" in t: return meeting_date, False
        m = re.search(r"\b(?:через|за|в течение) (\d+|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять) (день|дня|дней|неделю|недели|недель)\b", t)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else NUMBERS.get(m.group(1))
            if n: return meeting_date + timedelta(days=n * (7 if m.group(2).startswith("недел") else 1)), False
        for word, weekday in WEEKDAYS.items():
            if re.search(rf"\b(?:к|до) {word}\b", t):
                delta = (weekday - meeting_date.weekday()) % 7
                return (None, True) if delta == 0 else (meeting_date + timedelta(days=delta), False)
        return None, True

    def _dedupe(self, items: list[ActionItem]) -> list[ActionItem]:
        out: list[ActionItem] = []
        for item in items:
            duplicate = next((i for i, x in enumerate(out) if self._same(item.assignee, x.assignee) and SequenceMatcher(None, self._norm(item.task), self._norm(x.task)).ratio() >= 0.84), None)
            if duplicate is None:
                out.append(item)
            else:
                old = out[duplicate]
                if (bool(item.assignee) + bool(item.deadline_text), bool(item.deadline_iso), item.confidence) > (bool(old.assignee) + bool(old.deadline_text), bool(old.deadline_iso), old.confidence):
                    out[duplicate] = item
        return out

    def _grounded(self, task: str, evidence: str) -> bool:
        a, b = self._norm(task), self._norm(evidence)
        if a in b or SequenceMatcher(None, a, b).ratio() >= 0.28: return True
        return any(len(w) >= 5 and any(w[:5] == e[:5] for e in b.split()) for w in a.split())

    def _summary(self, raw: Any, items: list[ActionItem]) -> str:
        if isinstance(raw, str) and 0 < len(" ".join(raw.split())) <= 800: return " ".join(raw.split())
        if not items: return "Подтверждённые поручения не извлечены."
        return f"Подтверждённых поручений: {len(items)}; со сроком: {sum(x.deadline_text is not None for x in items)}."

    @staticmethod
    def _clean(value: Any) -> str | None:
        if value is None: return None
        value = " ".join(str(value).split()).strip()
        return value or None

    @staticmethod
    def _conf(value: Any, default: float) -> float:
        try: return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError): return default

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(WORDS.findall(text.casefold().replace("ё", "е")))

    def _contains(self, phrase: str, text: str) -> bool:
        return self._norm(phrase) in self._norm(text)

    def _same(self, a: str | None, b: str | None) -> bool:
        return a is None and b is None or bool(a and b and self._norm(a) == self._norm(b))
