"""Audio upload validation: extension, content sniffing, size, duration, hash."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from config.settings import Settings

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac",
    ".webm", ".mp4", ".mov", ".mpeg", ".mpga",
}


class AudioValidationError(ValueError):
    """Raised with a user-facing message when an upload cannot be accepted."""


@dataclass(frozen=True)
class AudioInfo:
    filename: str
    safe_filename: str
    extension: str
    size_bytes: int
    detected_format: str
    duration_secs: float | None
    sha256: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def safe_filename(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^\w\- ]", "_", stem, flags=re.UNICODE).strip(" ._") or "recording"
    ext = re.sub(r"[^a-z0-9]", "", ext.lower())
    return f"{stem[:80]}.{ext}" if ext else stem[:80]


def detect_format(head: bytes) -> str | None:
    """Recognise common audio/video containers by magic bytes."""
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mpeg-audio"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[4:8] == b"ftyp":
        return "mp4/m4a/mov"
    if head[:4] == b"OggS":
        return "ogg/opus"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm/matroska"
    return None


def probe_duration(data: bytes) -> float | None:
    try:
        import mutagen
        f = mutagen.File(io.BytesIO(data))
        if f is not None and f.info is not None and getattr(f.info, "length", None):
            return float(f.info.length)
    except Exception:  # noqa: BLE001 - duration is best-effort only
        pass
    return None


def inspect_audio(filename: str, data: bytes, settings: Settings) -> AudioInfo:
    ext = PurePath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported file type '{ext or filename}'. Allowed: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    if not data:
        raise AudioValidationError("The file is empty.")
    max_bytes = settings.max_audio_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise AudioValidationError(
            f"File is {len(data) / 1024 / 1024:.1f} MB; the limit is {settings.max_audio_mb} MB "
            "(change SIA_MAX_AUDIO_MB in .env if ElevenLabs allows more)."
        )
    fmt = detect_format(data[:16])
    if fmt is None:
        raise AudioValidationError(
            "The file content does not look like audio/video (unknown file signature). "
            "It may be corrupted or renamed."
        )
    duration = probe_duration(data)
    if duration is not None:
        if duration < 0.1:
            raise AudioValidationError("Audio is shorter than 0.1 second (ElevenLabs minimum).")
        if duration > settings.max_audio_minutes * 60:
            raise AudioValidationError(
                f"Recording is {duration / 60:.1f} minutes; the limit is "
                f"{settings.max_audio_minutes} minutes (SIA_MAX_AUDIO_MINUTES)."
            )
    return AudioInfo(
        filename=filename,
        safe_filename=safe_filename(filename),
        extension=ext,
        size_bytes=len(data),
        detected_format=fmt,
        duration_secs=duration,
        sha256=hashlib.sha256(data).hexdigest(),
    )
