"""Transcription orchestration: confirmation, no-resend guard, local caching."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config.settings import Settings
from database.db import Database
from schemas.transcript import Segment, Transcript
from services.elevenlabs_client import ElevenLabsClient, ElevenLabsError

SENTENCE_END = (".", "?", "!", "؟")


class TranscriptionError(Exception):
    """Base class for pre-flight failures (nothing was sent to ElevenLabs)."""


class ConfirmationRequired(TranscriptionError):
    pass


class ResendNotAllowed(TranscriptionError):
    pass


class AlreadyInProgress(TranscriptionError):
    pass


class LimitExceeded(TranscriptionError):
    pass


class ConfigMissing(TranscriptionError):
    pass


def parse_elevenlabs_response(raw: dict, gap_secs: float = 2.0, max_words: int = 120) -> Transcript:
    """Group word-level tokens into speaker segments, keeping original text."""
    words = raw.get("words") or []
    segments: list[Segment] = []
    cur: dict | None = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        text = re.sub(r"\s+", " ", "".join(cur["parts"])).strip()
        if text:
            segments.append(Segment(id=f"seg_{len(segments) + 1:04d}", speaker_id=cur["spk"],
                                    start=cur["start"], end=cur["end"], text=text))
        cur = None

    for w in words:
        wtype = w.get("type", "word")
        token = w.get("text", "")
        if wtype == "spacing":
            if cur is not None:
                cur["parts"].append(token)
            continue
        spk = w.get("speaker_id") or "speaker_unknown"
        start, end = w.get("start"), w.get("end")
        if cur is not None:
            gap = (start - cur["end"]) if (start is not None and cur["end"] is not None) else 0
            long_enough = cur["n"] >= max_words and cur["last"].endswith(SENTENCE_END)
            if spk != cur["spk"] or gap > gap_secs or long_enough:
                flush()
        if cur is None:
            cur = {"spk": spk, "start": start, "end": end, "parts": [], "n": 0, "last": ""}
        cur["parts"].append(token)
        cur["n"] += 1
        cur["last"] = token
        if end is not None:
            cur["end"] = end
    flush()

    if not segments and raw.get("text"):
        segments.append(Segment(id="seg_0001", speaker_id="speaker_unknown", text=raw["text"].strip()))

    duration = raw.get("audio_duration_secs")
    if duration is None and segments and segments[-1].end is not None:
        duration = segments[-1].end
    return Transcript(
        language_code=raw.get("language_code"),
        language_probability=raw.get("language_probability"),
        audio_duration_secs=duration,
        transcription_id=raw.get("transcription_id"),
        text=raw.get("text") or " ".join(s.text for s in segments),
        segments=segments,
    )


@dataclass
class TranscriptionTicket:
    meeting_id: str
    audio_path: Path
    language_code: str | None
    num_speakers: int | None


class TranscriptionService:
    def __init__(self, settings: Settings, db: Database,
                 client_factory: Callable[[], ElevenLabsClient] | None = None):
        self.settings, self.db = settings, db
        self._client_factory = client_factory or (lambda: ElevenLabsClient(
            settings.elevenlabs_api_key, settings.elevenlabs_base_url, settings.elevenlabs_stt_model))

    # ---- paths ---------------------------------------------------------
    def raw_path(self, meeting_id: str) -> Path:
        return self.settings.transcripts_dir / f"{meeting_id}.raw.json"

    def parsed_path(self, meeting_id: str) -> Path:
        return self.settings.transcripts_dir / f"{meeting_id}.json"

    # ---- step 1: pre-flight checks + atomic claim (synchronous) ----------
    def prepare(self, meeting_id: str, *, confirmed: bool, allow_resend: bool = False,
                language_code: str | None = None, num_speakers: int | None = None) -> TranscriptionTicket:
        if not confirmed:
            raise ConfirmationRequired("You must confirm that this request will use ElevenLabs credits.")
        missing = self.settings.missing_for_transcription()
        if missing:
            raise ConfigMissing("Missing in .env: " + ", ".join(missing))
        m = self.db.get_meeting(meeting_id)
        if m is None:
            raise TranscriptionError("Meeting not found.")
        status = m["transcription_status"]
        if status == "in_progress":
            raise AlreadyInProgress("A transcription request is already in progress for this meeting.")
        if status in ("done", "failed") and not allow_resend:
            raise ResendNotAllowed(
                f"This recording was already sent to ElevenLabs (status: {status}). "
                "Sending it again needs your explicit approval and uses credits again."
            )
        if m["audio_size_bytes"] > self.settings.max_audio_mb * 1024 * 1024:
            raise LimitExceeded(f"File exceeds {self.settings.max_audio_mb} MB.")
        dur = m["audio_duration_secs"]
        if dur and dur > self.settings.max_audio_minutes * 60:
            raise LimitExceeded(f"Recording exceeds {self.settings.max_audio_minutes} minutes.")
        audio = Path(m["audio_path"])
        if not audio.is_file():
            raise TranscriptionError("The saved recording file is missing on disk.")
        if not self.db.claim_transcription(meeting_id):
            raise AlreadyInProgress("A transcription request is already in progress for this meeting.")
        return TranscriptionTicket(meeting_id, audio, language_code, num_speakers)

    # ---- step 2: the paid call (may run in a background thread) -----------
    def execute(self, ticket: TranscriptionTicket) -> Transcript:
        mid = ticket.meeting_id
        try:
            raw = self._client_factory().transcribe(
                ticket.audio_path, language_code=ticket.language_code, num_speakers=ticket.num_speakers)
        except ElevenLabsError as exc:
            # No retry. The recording is untouched on disk.
            self.db.update_meeting(mid, transcription_status="failed", transcription_error=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self.db.update_meeting(mid, transcription_status="failed",
                                   transcription_error=f"Unexpected error: {type(exc).__name__}")
            raise
        # Save the raw response FIRST so a parsing bug can never cost extra credits.
        self._archive_existing(mid)
        self.settings.ensure_dirs()
        self.raw_path(mid).write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._parse_and_store(mid, raw)

    def request_transcription(self, meeting_id: str, **kwargs) -> Transcript:
        """Synchronous convenience: prepare + execute."""
        return self.execute(self.prepare(meeting_id, **kwargs))

    # ---- no-API recovery ---------------------------------------------------
    def rebuild_from_raw(self, meeting_id: str) -> Transcript:
        """Re-parse the saved raw response (costs nothing)."""
        p = self.raw_path(meeting_id)
        if not p.is_file():
            raise TranscriptionError("No saved ElevenLabs response exists for this meeting.")
        return self._parse_and_store(meeting_id, json.loads(p.read_text(encoding="utf-8")))

    def reset_interrupted(self, meeting_id: str) -> None:
        """User-confirmed: mark a stuck 'in_progress' request as failed (no API call)."""
        self.db.update_meeting(meeting_id, transcription_status="failed",
                               transcription_error="Interrupted (app closed during request). "
                               "Check your ElevenLabs usage before sending again.")

    def load_transcript(self, meeting_id: str) -> Transcript | None:
        p = self.parsed_path(meeting_id)
        if not p.is_file():
            return None
        return Transcript.model_validate_json(p.read_text(encoding="utf-8"))

    # ---- internals -----------------------------------------------------
    def _parse_and_store(self, meeting_id: str, raw: dict) -> Transcript:
        try:
            transcript = parse_elevenlabs_response(raw)
            if not transcript.segments:
                raise ValueError("no speech segments found")
        except Exception as exc:  # noqa: BLE001
            self.db.update_meeting(
                meeting_id, transcription_status="failed",
                transcription_error=f"Response saved but could not be parsed ({exc}). "
                "Use 'Rebuild from saved response' - it does not use credits.")
            raise TranscriptionError(str(exc)) from exc
        self.parsed_path(meeting_id).write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        self.db.update_meeting(meeting_id, transcription_status="done", transcription_error=None,
                               transcript_path=str(self.parsed_path(meeting_id)))
        return transcript

    def _archive_existing(self, meeting_id: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for p in (self.raw_path(meeting_id), self.parsed_path(meeting_id)):
            if p.is_file():
                p.rename(p.with_name(f"{p.stem}.{stamp}{p.suffix}"))
