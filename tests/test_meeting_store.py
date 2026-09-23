from __future__ import annotations

import os
import stat

from app.demo import demo_meeting_result
from app.services.meeting_store import list_meetings, load_meeting, save_meeting


def test_saved_meeting_survives_new_database_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("QAZMEETING_DB_PATH", str(tmp_path / "meetings.sqlite3"))
    original = demo_meeting_result("Тестовое совещание", None)

    identifier = save_meeting(original, "Демо-данные", "demo")
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "meetings.sqlite3").stat().st_mode) == 0o600
    listed = list_meetings()
    restored, source_label, source_kind = load_meeting(identifier)

    assert len(listed) == 1
    assert listed[0]["title"] == "Тестовое совещание"
    assert (source_label, source_kind) == ("Демо-данные", "demo")
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")

    changed = restored.model_copy(update={"summary": "Проверено вручную."})
    assert save_meeting(changed, source_label, source_kind, identifier) == identifier
    assert len(list_meetings()) == 1
    assert load_meeting(identifier)[0].summary == "Проверено вручную."
