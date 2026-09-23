"""Persistent local meeting results for the self-hosted interface."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models.schemas import MeetingResult


def _database_path() -> Path:
    return Path(os.getenv("QAZMEETING_DB_PATH", "data/meetings.sqlite3")).expanduser().resolve()


def _connect() -> sqlite3.Connection:
    path = _database_path()
    new_directory = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if new_directory and os.name == "posix":
        path.parent.chmod(0o700)
    connection = sqlite3.connect(path, timeout=30)
    if os.name == "posix":
        path.chmod(0o600)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_label TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def save_meeting(
    result: MeetingResult,
    source_label: str,
    source_kind: str,
    meeting_id: str | None = None,
) -> str:
    """Insert or update one validated MeetingResult and return its stable ID."""

    identifier = meeting_id or uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO meetings (id, title, source_label, source_kind, result_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                source_label=excluded.source_label,
                source_kind=excluded.source_kind,
                result_json=excluded.result_json,
                updated_at=excluded.updated_at
            """,
            (identifier, result.title, source_label, source_kind, payload, now, now),
        )
    return identifier


def list_meetings() -> list[dict[str, str]]:
    """List saved results without loading their transcripts into the interface."""

    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, title, source_label, source_kind, updated_at "
            "FROM meetings ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def load_meeting(meeting_id: str) -> tuple[MeetingResult, str, str]:
    with _connect() as connection:
        row = connection.execute(
            "SELECT result_json, source_label, source_kind FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Saved meeting not found: {meeting_id}")
    return MeetingResult.model_validate_json(row["result_json"]), row["source_label"], row["source_kind"]
