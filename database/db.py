"""SQLite persistence (stdlib sqlite3, one short-lived connection per call)."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    audio_filename TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    audio_sha256 TEXT NOT NULL,
    audio_size_bytes INTEGER NOT NULL,
    audio_duration_secs REAL,
    transcription_status TEXT NOT NULL DEFAULT 'not_started',
    transcription_attempts INTEGER NOT NULL DEFAULT 0,
    transcription_error TEXT,
    transcript_path TEXT,
    speakers_json TEXT,
    analysis_status TEXT NOT NULL DEFAULT 'not_started',
    analysis_started_at TEXT,
    analysis_error TEXT,
    report_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_meetings_sha ON meetings(audio_sha256);
"""

UPDATABLE = {
    "title", "transcription_status", "transcription_attempts", "transcription_error",
    "transcript_path", "speakers_json", "analysis_status", "analysis_started_at",
    "analysis_error", "report_path", "audio_duration_secs",
}
ANALYSIS_STALE_AFTER = timedelta(minutes=15)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---- meetings -------------------------------------------------------
    def create_meeting(self, *, title: str, audio_filename: str, audio_path: str,
                       audio_sha256: str, audio_size_bytes: int,
                       audio_duration_secs: float | None) -> str:
        meeting_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO meetings (id,title,created_at,audio_filename,audio_path,"
                "audio_sha256,audio_size_bytes,audio_duration_secs) VALUES (?,?,?,?,?,?,?,?)",
                (meeting_id, title, utcnow(), audio_filename, audio_path,
                 audio_sha256, audio_size_bytes, audio_duration_secs),
            )
        return meeting_id

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
        return dict(row) if row else None

    def list_meetings(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM meetings ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def find_by_sha256(self, sha: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM meetings WHERE audio_sha256=? ORDER BY created_at LIMIT 1", (sha,)
            ).fetchone()
        return dict(row) if row else None

    def update_meeting(self, meeting_id: str, **fields: Any) -> None:
        bad = set(fields) - UPDATABLE
        if bad:
            raise ValueError(f"Cannot update columns: {sorted(bad)}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE meetings SET {sets} WHERE id=?", (*fields.values(), meeting_id))

    def delete_meeting(self, meeting_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))

    # ---- atomic claims (prevent duplicate paid requests) -----------------
    def claim_transcription(self, meeting_id: str) -> bool:
        """Atomically move to 'in_progress'. False if already in progress."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE meetings SET transcription_status='in_progress', "
                "transcription_attempts=transcription_attempts+1, transcription_error=NULL "
                "WHERE id=? AND transcription_status!='in_progress'",
                (meeting_id,),
            )
            return cur.rowcount == 1

    def claim_analysis(self, meeting_id: str) -> bool:
        """Atomically move to 'in_progress'; a stale lock (>15 min) can be taken over."""
        stale = (datetime.now(timezone.utc) - ANALYSIS_STALE_AFTER).isoformat(timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                "UPDATE meetings SET analysis_status='in_progress', analysis_started_at=?, "
                "analysis_error=NULL WHERE id=? AND (analysis_status!='in_progress' "
                "OR analysis_started_at IS NULL OR analysis_started_at<?)",
                (utcnow(), meeting_id, stale),
            )
            return cur.rowcount == 1
