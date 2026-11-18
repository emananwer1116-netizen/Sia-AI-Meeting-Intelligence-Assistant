"""Structured meeting report. Every item carries evidence from the transcript."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

UNASSIGNED = "Unassigned"


class ItemStatus(str, Enum):
    CONFIRMED = "Confirmed"
    NEEDS_REVIEW = "Needs Review"
    UNASSIGNED = "Unassigned"


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Evidence(BaseModel):
    segment_id: str
    quote: str                      # key phrase, verified to exist in the segment
    speaker: str = ""
    speaker_id: str = ""
    timestamp: float | None = None
    segment_text: str = ""          # full original segment text, untouched


class _Item(BaseModel):
    id: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class Decision(_Item):
    text: str = ""
    status: ItemStatus = ItemStatus.CONFIRMED
    confidence: Confidence = Confidence.MEDIUM


class ProposedIdea(_Item):
    text: str = ""
    proposed_by: str | None = None


class ActionItem(_Item):
    description: str = ""
    assigned_to: str = UNASSIGNED
    assigned_by: str | None = None
    kind: str = "mentor_instruction"      # or "student_commitment"
    deadline: str | None = None           # verbatim words as spoken
    status: ItemStatus = ItemStatus.NEEDS_REVIEW
    confidence: Confidence = Confidence.MEDIUM
    depends_on: list[str] = Field(default_factory=list)
    user_reviewed: bool = False

    @property
    def timestamp(self) -> float | None:
        return self.evidence[0].timestamp if self.evidence else None


class Dependency(_Item):
    task_id: str = ""
    depends_on_task_id: str = ""
    description: str = ""


class Blocker(_Item):
    description: str = ""
    raised_by: str | None = None


class UnresolvedQuestion(_Item):
    question: str = ""
    raised_by: str | None = None


class MentorInstruction(_Item):
    text: str = ""
    addressed_to: str | None = None


class Conflict(_Item):
    description: str = ""


class RejectedItem(BaseModel):
    section: str
    text: str
    reason: str


class MeetingReport(BaseModel):
    summary: str = ""
    goals: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    proposed_ideas: list[ProposedIdea] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    unresolved_questions: list[UnresolvedQuestion] = Field(default_factory=list)
    mentor_instructions: list[MentorInstruction] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    validator_notes: list[str] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)

    def tasks_by_person(self) -> dict[str, list[ActionItem]]:
        groups: dict[str, list[ActionItem]] = {}
        for item in self.action_items:
            groups.setdefault(item.assigned_to or UNASSIGNED, []).append(item)
        # Unassigned last
        ordered = {k: v for k, v in groups.items() if k != UNASSIGNED}
        if UNASSIGNED in groups:
            ordered[UNASSIGNED] = groups[UNASSIGNED]
        return ordered
