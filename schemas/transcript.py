"""Transcript and speaker schemas."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    MENTOR = "Mentor"
    STUDENT = "Student"
    TEAM_LEAD = "Team Lead"
    OTHER = "Other"
    UNKNOWN = "Unknown"


def format_ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    s = int(max(seconds, 0))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def friendly_speaker_label(speaker_id: str) -> str:
    """'speaker_0' -> 'Speaker 0'."""
    if speaker_id.startswith("speaker_"):
        return "Speaker " + speaker_id.split("_", 1)[1]
    return speaker_id.replace("_", " ").title()


class Segment(BaseModel):
    id: str
    speaker_id: str
    start: float | None = None
    end: float | None = None
    text: str


class Transcript(BaseModel):
    language_code: str | None = None
    language_probability: float | None = None
    audio_duration_secs: float | None = None
    transcription_id: str | None = None
    text: str = ""
    segments: list[Segment] = Field(default_factory=list)

    def speaker_ids(self) -> list[str]:
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker_id not in seen:
                seen.append(seg.speaker_id)
        return seen

    def segment_by_id(self, segment_id: str) -> Segment | None:
        for seg in self.segments:
            if seg.id == segment_id:
                return seg
        return None


class SpeakerInfo(BaseModel):
    speaker_id: str
    name: str = ""
    role: SpeakerRole = SpeakerRole.UNKNOWN
    # Alternative spellings (e.g. Arabic form of an English name) used only to
    # check that a person was really named in the evidence.
    aliases: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name.strip() or friendly_speaker_label(self.speaker_id)


class SpeakerMap(BaseModel):
    speakers: list[SpeakerInfo] = Field(default_factory=list)

    @property
    def mentor(self) -> SpeakerInfo | None:
        for sp in self.speakers:
            if sp.role == SpeakerRole.MENTOR:
                return sp
        return None

    def get(self, speaker_id: str) -> SpeakerInfo | None:
        for sp in self.speakers:
            if sp.speaker_id == speaker_id:
                return sp
        return None

    def display_name(self, speaker_id: str) -> str:
        sp = self.get(speaker_id)
        return sp.display_name if sp else friendly_speaker_label(speaker_id)

    def problems(self, known_speaker_ids: list[str] | None = None) -> list[str]:
        """Return human-readable problems; empty list means ready for analysis."""
        errors: list[str] = []
        mentors = [s for s in self.speakers if s.role == SpeakerRole.MENTOR]
        if len(mentors) == 0:
            errors.append("No Mentor selected.")
        elif len(mentors) > 1:
            errors.append("Only one speaker can be the Mentor.")
        names = [s.display_name.casefold() for s in self.speakers]
        if len(set(names)) != len(names):
            errors.append("Two speakers have the same name.")
        if known_speaker_ids is not None:
            missing = [sid for sid in known_speaker_ids if self.get(sid) is None]
            if missing:
                errors.append("Speakers not defined: " + ", ".join(missing))
        return errors
