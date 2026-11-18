"""Meeting analysis: prompt building + strict post-validation of the LLM output.

The LLM only *proposes*. Nothing reaches the report unless its evidence quote is
found in the cited transcript segment, and owners/deadlines are checked against
the evidence text. That is what prevents invented tasks, owners and deadlines.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from schemas.meeting_report import (
    UNASSIGNED, ActionItem, Blocker, Confidence, Conflict, Decision, Dependency, Evidence,
    ItemStatus, MeetingReport, MentorInstruction, ProposedIdea, RejectedItem, UnresolvedQuestion,
)
from schemas.transcript import SpeakerMap, SpeakerRole, Transcript, format_ts
from services.litellm_client import LiteLLMClient
from services.text_utils import name_in_text, normalize, quote_in_text


class AnalysisError(Exception):
    pass


SYSTEM_PROMPT = """You are the analysis engine of "Sia", an AI meeting assistant. You receive a transcript of a meeting between a Mentor and team members. Language: Egyptian Arabic mixed with English technical terms.

HARD RULES
1. Use ONLY what is in the transcript. Never invent tasks, decisions, names, owners or deadlines.
2. Classify carefully:
   - decisions: explicitly agreed/decided. A suggestion that was not adopted ("ممكن نضيف...", "what if...") is a proposed_idea, NEVER a decision.
   - action_items: kind "mentor_instruction" = the Mentor explicitly asks/assigns something; kind "student_commitment" = a participant explicitly commits ("هعمل", "I will").
   - Ordinary discussion or a student's remark is NOT a task. Not every Mentor sentence is a task for a specific person.
   - blockers, unresolved_questions (asked but not settled), mentor_instructions (a standalone list of the Mentor's directions), conflicts (contradicting instructions: list both, do not pick one).
3. assigned_to: a participant display name ONLY if the transcript clearly shows who owns the task (named or directly addressed). If no person is identified (e.g. "محتاجين نراجع الـ API") use "Unassigned". Never guess.
4. deadline: only if explicitly spoken. Copy the words exactly as spoken (e.g. "قبل يوم الخميس"). Otherwise null. Never convert to a date, never infer.
5. EVERY item needs evidence: a list of {"segment_id": "...", "quote": "..."}. segment_id must be one of the ids shown in the transcript. quote must be copied VERBATIM, character for character, from that segment (a clause or sentence is enough). Never translate or rephrase quotes.
6. If you are unsure, lower the confidence and explain in review_note.
7. Write summary, goals and scope in the meeting's language (keep English technical terms as spoken). Be concise.
8. Respond with ONE JSON object only. No markdown fences, no commentary.

JSON SHAPE
{
 "summary": "string", "goals": ["string"], "scope": ["string"],
 "decisions": [{"id":"D1","text":"","confidence":"High|Medium|Low","review_note":null,"evidence":[{"segment_id":"seg_0001","quote":""}]}],
 "proposed_ideas": [{"id":"I1","text":"","proposed_by":"name or null","evidence":[...]}],
 "action_items": [{"id":"T1","description":"","kind":"mentor_instruction|student_commitment","assigned_to":"name or Unassigned","assigned_by":"mentor name or null","deadline":"words as spoken or null","confidence":"High|Medium|Low","review_note":null,"depends_on":["T2"],"evidence":[...]}],
 "dependencies": [{"task_id":"T2","depends_on_task_id":"T1","description":"","evidence":[...]}],
 "blockers": [{"id":"B1","description":"","raised_by":"name or null","evidence":[...]}],
 "unresolved_questions": [{"id":"Q1","question":"","raised_by":"name or null","evidence":[...]}],
 "mentor_instructions": [{"id":"M1","text":"","addressed_to":"name, Unassigned or Everyone","evidence":[...]}],
 "conflicts": [{"id":"C1","description":"","evidence":[...]}]
}
Use empty lists when nothing applies."""


def build_transcript_text(transcript: Transcript, speakers: SpeakerMap) -> str:
    lines = []
    for seg in transcript.segments:
        sp = speakers.get(seg.speaker_id)
        who = sp.display_name if sp else seg.speaker_id
        role = f" ({sp.role.value})" if sp and sp.role != SpeakerRole.UNKNOWN else ""
        lines.append(f"[{seg.id}] [{format_ts(seg.start)}] {who}{role}: {seg.text}")
    return "\n".join(lines)


def build_messages(transcript: Transcript, speakers: SpeakerMap) -> list[dict[str, str]]:
    roster = "\n".join(
        f"- {s.display_name} — role: {s.role.value}"
        + (f" (also written as: {', '.join(s.aliases)})" if s.aliases else "")
        for s in speakers.speakers
    )
    mentor = speakers.mentor
    user = (
        f"PARTICIPANTS (roles were set manually by the user; do not guess identities):\n{roster}\n"
        f"The Mentor is: {mentor.display_name if mentor else 'unknown'}\n\n"
        f"TRANSCRIPT:\n{build_transcript_text(transcript, speakers)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_llm_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise AnalysisError("The model did not return a JSON object.")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AnalysisError("The model JSON is not an object.")
    return data


# ---- lenient raw models (LLM output) ------------------------------------
class _RawEvidence(BaseModel):
    segment_id: str = ""
    quote: str = ""


class _RawItem(BaseModel):
    model_config = {"extra": "ignore"}
    id: str | None = None
    evidence: list[_RawEvidence] = Field(default_factory=list)
    review_note: str | None = None
    confidence: str | None = None
    text: str | None = None
    description: str | None = None
    question: str | None = None
    kind: str | None = None
    assigned_to: str | None = None
    assigned_by: str | None = None
    deadline: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    proposed_by: str | None = None
    raised_by: str | None = None
    addressed_to: str | None = None
    task_id: str | None = None
    depends_on_task_id: str | None = None


class _RawReport(BaseModel):
    model_config = {"extra": "ignore"}
    summary: str = ""
    goals: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    decisions: list[_RawItem] = Field(default_factory=list)
    proposed_ideas: list[_RawItem] = Field(default_factory=list)
    action_items: list[_RawItem] = Field(default_factory=list)
    dependencies: list[_RawItem] = Field(default_factory=list)
    blockers: list[_RawItem] = Field(default_factory=list)
    unresolved_questions: list[_RawItem] = Field(default_factory=list)
    mentor_instructions: list[_RawItem] = Field(default_factory=list)
    conflicts: list[_RawItem] = Field(default_factory=list)


def _conf(value: str | None) -> Confidence:
    for c in Confidence:
        if (value or "").strip().lower() == c.value.lower():
            return c
    return Confidence.MEDIUM


_UNASSIGNED_WORDS = {"", "unassigned", "none", "null", "n/a", "unknown", "غير محدد", "لم يحدد", "بدون"}


class _Validator:
    def __init__(self, transcript: Transcript, speakers: SpeakerMap):
        self.t, self.sp = transcript, speakers
        self.mentor = speakers.mentor
        self.rejected: list[RejectedItem] = []

    # ---- evidence ---------------------------------------------------------
    def evidence(self, raw: list[_RawEvidence]) -> tuple[list[Evidence], list[str]]:
        out: list[Evidence] = []
        notes: list[str] = []
        for ev in raw:
            quote = (ev.quote or "").strip()
            seg = self.t.segment_by_id(ev.segment_id)
            if not quote:
                continue
            if seg is None or not quote_in_text(quote, seg.text):
                alt = next((s for s in self.t.segments if quote_in_text(quote, s.text)), None)
                if alt is None:
                    notes.append(f"Evidence quote not found in transcript and was discarded: “{quote[:70]}”")
                    continue
                seg = alt
            if any(e.segment_id == seg.id and e.quote == quote for e in out):
                continue
            out.append(Evidence(segment_id=seg.id, quote=quote, speaker=self.sp.display_name(seg.speaker_id),
                                speaker_id=seg.speaker_id, timestamp=seg.start, segment_text=seg.text))
        return out, notes

    def reject(self, section: str, text: str, reason: str) -> None:
        self.rejected.append(RejectedItem(section=section, text=(text or "")[:300], reason=reason))

    # ---- people -----------------------------------------------------------
    def resolve_person(self, name: str | None) -> tuple[str, bool]:
        """Return (canonical name, is_known_participant)."""
        if (name or "").strip().casefold() in _UNASSIGNED_WORDS:
            return UNASSIGNED, True
        n = normalize(name or "")
        for s in self.sp.speakers:
            if n == normalize(s.display_name) or n in {normalize(a) for a in s.aliases}:
                return s.display_name, True
        return (name or "").strip(), False

    def person_named_in(self, person: str, evs: list[Evidence]) -> bool:
        info = next((s for s in self.sp.speakers if s.display_name == person), None)
        candidates = [person] + (info.aliases if info else [])
        texts = [e.quote for e in evs] + [e.segment_text for e in evs if self.is_mentor(e.speaker_id)]
        return any(name_in_text(c, t) for c in candidates for t in texts)

    def is_mentor(self, speaker_id: str) -> bool:
        return self.mentor is not None and speaker_id == self.mentor.speaker_id

    # ---- builders -----------------------------------------------------------
    def build(self, raw: _RawReport) -> MeetingReport:
        rep = MeetingReport(summary=raw.summary.strip(), goals=[g for g in raw.goals if g.strip()],
                            scope=[s for s in raw.scope if s.strip()])
        for r in raw.decisions:
            evs, notes = self.evidence(r.evidence)
            if not evs:
                self.reject("decisions", r.text or "", "no verifiable evidence")
                continue
            conf = _conf(r.confidence)
            if r.review_note:
                notes.append(r.review_note)
            speakers_in_ev = {e.speaker_id for e in evs}
            if not any(self.is_mentor(s) for s in speakers_in_ev) and len(speakers_in_ev) < 2:
                notes.append("Backed only by one non-Mentor speaker; this may be a proposal, not a decision.")
            review = bool(notes) or conf == Confidence.LOW
            rep.decisions.append(Decision(id=f"D{len(rep.decisions) + 1}", text=(r.text or "").strip(),
                                          evidence=evs, review_notes=notes, confidence=conf,
                                          status=ItemStatus.NEEDS_REVIEW if review else ItemStatus.CONFIRMED))
        for r in raw.proposed_ideas:
            evs, notes = self.evidence(r.evidence)
            if not evs:
                self.reject("proposed_ideas", r.text or "", "no verifiable evidence")
                continue
            by, _ = self.resolve_person(r.proposed_by)
            rep.proposed_ideas.append(ProposedIdea(id=f"I{len(rep.proposed_ideas) + 1}", text=(r.text or "").strip(),
                                                   proposed_by=None if by == UNASSIGNED else by,
                                                   evidence=evs, review_notes=notes))
        id_map: dict[str, str] = {}
        for r in raw.action_items:
            item = self._action(r, len(rep.action_items) + 1)
            if item is None:
                continue
            if r.id:
                id_map[r.id.strip()] = item.id
            rep.action_items.append(item)
        self._link_dependencies(raw, rep, id_map)
        for r in raw.blockers:
            evs, notes = self.evidence(r.evidence)
            if not evs:
                self.reject("blockers", r.description or r.text or "", "no verifiable evidence")
                continue
            by, _ = self.resolve_person(r.raised_by)
            rep.blockers.append(Blocker(id=f"B{len(rep.blockers) + 1}", description=(r.description or r.text or "").strip(),
                                        raised_by=None if by == UNASSIGNED else by, evidence=evs, review_notes=notes))
        for r in raw.unresolved_questions:
            evs, notes = self.evidence(r.evidence)
            if not evs:
                self.reject("unresolved_questions", r.question or r.text or "", "no verifiable evidence")
                continue
            by, _ = self.resolve_person(r.raised_by)
            rep.unresolved_questions.append(UnresolvedQuestion(
                id=f"Q{len(rep.unresolved_questions) + 1}", question=(r.question or r.text or "").strip(),
                raised_by=None if by == UNASSIGNED else by, evidence=evs, review_notes=notes))
        for r in raw.mentor_instructions:
            evs, notes = self.evidence(r.evidence)
            if not evs or not any(self.is_mentor(e.speaker_id) for e in evs):
                self.reject("mentor_instructions", r.text or "", "no verifiable evidence spoken by the Mentor")
                continue
            to, known = self.resolve_person(r.addressed_to)
            if (r.addressed_to or "").strip().casefold() == "everyone":
                to, known = "Everyone", True
            if not known:
                notes.append(f"'{to}' is not a recorded participant.")
            rep.mentor_instructions.append(MentorInstruction(
                id=f"M{len(rep.mentor_instructions) + 1}", text=(r.text or "").strip(), addressed_to=to,
                evidence=evs, review_notes=notes))
        for r in raw.conflicts:
            evs, notes = self.evidence(r.evidence)
            if not evs:
                self.reject("conflicts", r.description or r.text or "", "no verifiable evidence")
                continue
            rep.conflicts.append(Conflict(id=f"C{len(rep.conflicts) + 1}", description=(r.description or r.text or "").strip(),
                                          evidence=evs, review_notes=notes))
        if self.rejected:
            rep.validator_notes.append(
                f"{len(self.rejected)} item(s) proposed by the model were rejected because they could not be "
                "tied to the transcript (see 'Rejected by validator').")
        rep.rejected_items = self.rejected
        return rep

    def _action(self, r: _RawItem, n: int) -> ActionItem | None:
        desc = (r.description or r.text or "").strip()
        evs, notes = self.evidence(r.evidence)
        if not evs:
            self.reject("action_items", desc, "no verifiable evidence")
            return None
        kind = "student_commitment" if (r.kind or "").strip().lower().startswith("student") else "mentor_instruction"
        assignee, known = self.resolve_person(r.assigned_to)
        if r.review_note:
            notes.append(r.review_note)
        assigned_by: str | None = None
        if kind == "mentor_instruction":
            if self.mentor and any(self.is_mentor(e.speaker_id) for e in evs):
                assigned_by = self.mentor.display_name
            else:
                notes.append("No evidence spoken by the Mentor; confirm this is really a Mentor instruction.")
        if assignee != UNASSIGNED:
            if not known:
                notes.append(f"'{assignee}' is not a recorded participant.")
            elif kind == "mentor_instruction" and not self.person_named_in(assignee, evs):
                notes.append(f"{assignee} is not named in the evidence; the owner may have been inferred from context.")
            elif kind == "student_commitment" and not any(
                    self.sp.display_name(e.speaker_id) == assignee for e in evs):
                notes.append(f"The commitment is not spoken by {assignee}.")
        deadline = (r.deadline or "").strip() or None
        if deadline and normalize(deadline) in {"null", "none", "n a"}:
            deadline = None
        if deadline:
            pool = [e.quote for e in evs] + [e.segment_text for e in evs]
            if not any(quote_in_text(deadline, t, min_len=2) for t in pool):
                notes.append(f"Deadline “{deadline}” was not found in the evidence and was removed.")
                deadline = None
        conf = _conf(r.confidence)
        if assignee == UNASSIGNED:
            status = ItemStatus.UNASSIGNED
        elif notes or conf == Confidence.LOW:
            status = ItemStatus.NEEDS_REVIEW
        else:
            status = ItemStatus.CONFIRMED
        return ActionItem(id=f"T{n}", description=desc, assigned_to=assignee, assigned_by=assigned_by,
                          kind=kind, deadline=deadline, status=status, confidence=conf, evidence=evs,
                          review_notes=notes, depends_on=[x for x in r.depends_on])

    def _link_dependencies(self, raw: _RawReport, rep: MeetingReport, id_map: dict[str, str]) -> None:
        valid = {a.id for a in rep.action_items}
        # remap depends_on from model ids to our ids; drop unknown references
        for a in rep.action_items:
            a.depends_on = [id_map[d.strip()] for d in a.depends_on if d.strip() in id_map and id_map[d.strip()] != a.id]
        for r in raw.dependencies:
            tid, did = id_map.get((r.task_id or "").strip()), id_map.get((r.depends_on_task_id or "").strip())
            evs, notes = self.evidence(r.evidence)
            if tid not in valid or did not in valid or tid == did:
                self.reject("dependencies", r.description or r.text or "", "refers to an unknown task")
                continue
            if not evs:
                self.reject("dependencies", r.description or r.text or "", "no verifiable evidence")
                continue
            rep.dependencies.append(Dependency(id=f"X{len(rep.dependencies) + 1}", task_id=tid, depends_on_task_id=did,
                                               description=(r.description or r.text or "").strip(),
                                               evidence=evs, review_notes=notes))


def build_report(raw: dict, transcript: Transcript, speakers: SpeakerMap) -> MeetingReport:
    try:
        parsed = _RawReport.model_validate(raw)
    except ValidationError as exc:
        raise AnalysisError(f"The model JSON does not match the expected structure: {exc.errors()[0]['loc']}") from exc
    return _Validator(transcript, speakers).build(parsed)


class MeetingAnalyzer:
    def __init__(self, llm: LiteLLMClient, max_prompt_chars: int = 150_000):
        self.llm, self.max_prompt_chars = llm, max_prompt_chars

    def request_raw(self, transcript: Transcript, speakers: SpeakerMap) -> str:
        """Pre-flight checks + exactly one LLM request. Returns the raw model text."""
        problems = speakers.problems(transcript.speaker_ids())
        if problems:
            raise AnalysisError("Speaker setup incomplete: " + " ".join(problems))
        messages = build_messages(transcript, speakers)
        size = sum(len(m["content"]) for m in messages)
        if size > self.max_prompt_chars:
            raise AnalysisError(
                f"Transcript is too long for a single request ({size:,} characters; limit "
                f"{self.max_prompt_chars:,}). Chunked analysis is not part of this MVP.")
        return self.llm.chat(messages)

    @staticmethod
    def build(raw_text: str, transcript: Transcript, speakers: SpeakerMap) -> MeetingReport:
        return build_report(parse_llm_json(raw_text), transcript, speakers)

    def analyze(self, transcript: Transcript, speakers: SpeakerMap) -> tuple[MeetingReport, str]:
        raw_text = self.request_raw(transcript, speakers)
        return self.build(raw_text, transcript, speakers), raw_text
