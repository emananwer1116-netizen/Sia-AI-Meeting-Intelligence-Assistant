"""Permanent deletion of a meeting (recording, transcript files, reports, DB row)."""
from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from database.db import Database


class DeleteNotAllowed(Exception):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return path.resolve() != root.resolve()
    except ValueError:
        return False


def delete_meeting(settings: Settings, db: Database, meeting_id: str) -> int:
    """Delete everything stored for a meeting. Returns the number of files removed.

    Files are removed first and the DB row last, so a failure never leaves a
    meeting that looks intact but has lost its files silently.
    """
    m = db.get_meeting(meeting_id)
    if m is None:
        raise DeleteNotAllowed("Meeting not found.")
    if "in_progress" in (m["transcription_status"], m["analysis_status"]):
        raise DeleteNotAllowed("A request is still running for this meeting. Wait until it finishes.")
    targets: list[Path] = []
    audio = Path(m["audio_path"])
    if _inside(audio, settings.recordings_dir):          # never touch files outside storage/recordings
        targets.append(audio)
    for folder in (settings.transcripts_dir, settings.reports_dir):
        targets += [p for p in folder.glob(f"{meeting_id}.*") if p.is_file()]
    removed = 0
    try:
        for p in targets:
            if p.exists():
                p.unlink()
                removed += 1
        parent = audio.parent
        if _inside(parent, settings.recordings_dir) and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError as exc:
        raise DeleteNotAllowed(f"Could not delete a file (is it open in another program?): {exc}") from exc
    db.delete_meeting(meeting_id)
    return removed
