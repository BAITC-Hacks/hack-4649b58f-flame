from datetime import date
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from app.agents.meeting_agent import MeetingProtocolAgent, ModelResponseError
from app.config import local_model_url
from app.models.schemas import TranscriptSegment


def test_agent_honors_runtime_settings(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    agent = MeetingProtocolAgent()
    assert agent.timeout == 240
    assert agent.num_ctx == 8192
    assert MeetingProtocolAgent(timeout=7).timeout == 7


@pytest.mark.parametrize("invalid_id", [[], {}, True, 1.0])
def test_malformed_candidate_does_not_discard_valid_actions(invalid_id):
    source = [TranscriptSegment(id=1, start=0, end=2, speaker_id="S1",
                                text="Ерлан, подготовь отчёт.")]
    valid = {"task": "Подготовить отчёт", "assignee": "Ерлан",
             "author_speaker_id": "S1", "evidence_segment_id": 1, "confidence": 0.9}

    class StubAgent(MeetingProtocolAgent):
        def _call_ollama(self, prompt):
            return json.dumps({"speakers": [], "action_items": [
                {**valid, "evidence_segment_id": invalid_id}, valid,
            ]})

    result = StubAgent().run(source, date(2026, 9, 21))
    assert len(result.action_items) == 1
    assert result.transcript == source
    assert any("evidence_segment_id" in warning for warning in result.warnings)


@pytest.mark.parametrize("deadline,meeting", [
    ("через 999999999999 дней", date(2026, 9, 21)),
    ("999999999999 апта ішінде", date(2026, 9, 21)),
    ("завтра", date.max),
    ("к среде", date(9999, 12, 30)),
])
def test_unrepresentable_deadline_is_ambiguous(deadline, meeting):
    assert MeetingProtocolAgent()._deadline(deadline, meeting) == (None, True)


def test_invalid_deadline_does_not_discard_other_actions():
    source = [
        TranscriptSegment(id=1, start=0, end=2, speaker_id="S1",
                          text="Ерлан, подготовь отчёт через 999999999999 дней."),
        TranscriptSegment(id=2, start=2, end=4, speaker_id="S1",
                          text="Ботагоз, проверь договор."),
    ]

    class StubAgent(MeetingProtocolAgent):
        def _call_ollama(self, prompt):
            return json.dumps({"speakers": [], "action_items": [
                {"task": "Подготовить отчёт", "assignee": "Ерлан",
                 "deadline_text": "через 999999999999 дней", "evidence_segment_id": 1},
                {"task": "Проверить договор", "assignee": "Ботагоз", "evidence_segment_id": 2},
            ]})

    result = StubAgent().run(source, date(2026, 9, 21))
    assert len(result.action_items) == 2
    assert result.action_items[0].deadline_iso is None
    assert result.action_items[0].needs_review
    assert result.action_items[0].deadline_text == "через 999999999999 дней"


@pytest.mark.parametrize("model_deadline", [None, "срок не указан"])
def test_recovered_negated_deadline_is_not_normalized(model_deadline):
    agent = MeetingProtocolAgent()
    source = [TranscriptSegment(id=1, start=0, end=2, speaker_id="S1",
                                text="Ерлан, подготовь отчёт, но не к пятнице.")]
    actions = agent._actions([{
        "task": "Подготовить отчёт", "assignee": "Ерлан",
        "author_speaker_id": "S1", "deadline_text": model_deadline,
        "evidence_segment_id": 1, "confidence": 0.9,
    }], source, date(2026, 9, 21), [])
    assert len(actions) == 1
    assert actions[0].deadline_iso is None
    assert actions[0].needs_review


def test_iso_date_is_one_deadline_candidate():
    agent = MeetingProtocolAgent()
    assert agent._deadline_candidates("Подготовить отчёт к 2026-09-25.") == ["2026-09-25"]


def test_exact_deadline_does_not_match_inside_another_word():
    assert not MeetingProtocolAgent()._contains_exact_phrase("завтра", "послезавтра")


@pytest.mark.parametrize("url", [
    "http://localhost:11434@external.example",
    "http://localhost:invalid",
    "http://localhost:11434?target=external",
    "http://localhost:11434#fragment",
])
def test_invalid_endpoint_rejected_by_config_and_agent(monkeypatch, url):
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    with pytest.raises(ValueError):
        local_model_url()
    with pytest.raises(ValueError):
        MeetingProtocolAgent(base_url=url)


def test_ipv6_loopback_works_through_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://[::1]:11434")
    assert MeetingProtocolAgent().base_url == "http://[::1]:11434"


@pytest.mark.parametrize("status", [200, 301, 302, 303, 307, 308])
def test_transport_response_and_redirect_policy(status):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(b'{"message":{"content":"{}"}}')

        def do_GET(self):
            received.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"message":{"content":"{}"}}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = MeetingProtocolAgent(base_url=f"http://127.0.0.1:{server.server_port}")
        if status == 200:
            assert agent._call_ollama("synthetic test") == "{}"
        else:
            with pytest.raises(ModelResponseError):
                agent._call_ollama("synthetic test")
        assert received == ["/api/chat"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
