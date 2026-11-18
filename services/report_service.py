"""Analysis orchestration, report persistence and Markdown export."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from config.settings import Settings
from database.db import Database
from schemas.meeting_report import UNASSIGNED, Evidence, MeetingReport
from schemas.transcript import SpeakerMap, Transcript, format_ts
from services.litellm_client import LiteLLMClient
from services.meeting_analyzer import AnalysisError, MeetingAnalyzer


class AnalysisNotAllowed(Exception):
    """Pre-flight failure: no LLM request was made."""


class ReportService:
    def __init__(self, settings: Settings, db: Database,
                 llm_factory: Callable[[], LiteLLMClient] | None = None):
        self.settings, self.db = settings, db
        self._llm_factory = llm_factory or (lambda: LiteLLMClient(
            settings.litellm_base_url, settings.litellm_api_key, settings.litellm_model,
            timeout=settings.litellm_timeout_secs, json_mode=settings.litellm_json_mode,
            max_tokens=settings.litellm_max_tokens))

    # ---- paths ---------------------------------------------------------
    def report_path(self, mid: str) -> Path:
        return self.settings.reports_dir / f"{mid}.json"

    def original_path(self, mid: str) -> Path:
        return self.settings.reports_dir / f"{mid}.original.json"

    def llm_raw_path(self, mid: str) -> Path:
        return self.settings.reports_dir / f"{mid}.llm_raw.txt"

    # ---- speakers ------------------------------------------------------
    def save_speakers(self, mid: str, speakers: SpeakerMap, transcript: Transcript) -> list[str]:
        problems = speakers.problems(transcript.speaker_ids())
        if problems:
            return problems
        self.db.update_meeting(mid, speakers_json=speakers.model_dump_json())
        return []

    def load_speakers(self, mid: str) -> SpeakerMap | None:
        m = self.db.get_meeting(mid)
        if not m or not m["speakers_json"]:
            return None
        return SpeakerMap.model_validate_json(m["speakers_json"])

    # ---- analysis --------------------------------------------------------
    def prepare_analysis(self, mid: str, transcript: Transcript | None, *, allow_reanalyze: bool = False) -> SpeakerMap:
        missing = self.settings.missing_for_analysis()
        if missing:
            raise AnalysisNotAllowed("Missing in .env: " + ", ".join(missing))
        m = self.db.get_meeting(mid)
        if m is None or transcript is None:
            raise AnalysisNotAllowed("A transcript is required before analysis.")
        speakers = self.load_speakers(mid)
        if speakers is None:
            raise AnalysisNotAllowed("Save the speaker identification first.")
        problems = speakers.problems(transcript.speaker_ids())
        if problems:
            raise AnalysisNotAllowed(" ".join(problems))
        if m["analysis_status"] == "done" and not allow_reanalyze:
            raise AnalysisNotAllowed("This meeting was already analyzed. Re-analysis needs explicit approval.")
        if not self.db.claim_analysis(mid):
            raise AnalysisNotAllowed("An analysis is already running for this meeting.")
        return speakers

    def execute_analysis(self, mid: str, transcript: Transcript, speakers: SpeakerMap) -> MeetingReport:
        analyzer = MeetingAnalyzer(self._llm_factory(), self.settings.max_prompt_chars)
        try:
            raw_text = analyzer.request_raw(transcript, speakers)
            self.settings.ensure_dirs()
            self._archive(mid)
            self.llm_raw_path(mid).write_text(raw_text, encoding="utf-8")  # keep before parsing
            report = analyzer.build(raw_text, transcript, speakers)
        except Exception as exc:  # noqa: BLE001
            self.db.update_meeting(mid, analysis_status="failed", analysis_error=str(exc) or type(exc).__name__)
            raise
        self.original_path(mid).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        self.report_path(mid).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        self.db.update_meeting(mid, analysis_status="done", analysis_error=None, report_path=str(self.report_path(mid)))
        return report

    def run_analysis(self, mid: str, transcript: Transcript, *, allow_reanalyze: bool = False) -> MeetingReport:
        speakers = self.prepare_analysis(mid, transcript, allow_reanalyze=allow_reanalyze)
        return self.execute_analysis(mid, transcript, speakers)

    def reset_interrupted_analysis(self, mid: str) -> None:
        self.db.update_meeting(mid, analysis_status="failed", analysis_error="Interrupted before finishing.")

    # ---- report io ---------------------------------------------------------
    def load_report(self, mid: str) -> MeetingReport | None:
        p = self.report_path(mid)
        return MeetingReport.model_validate_json(p.read_text(encoding="utf-8")) if p.is_file() else None

    def save_report(self, mid: str, report: MeetingReport) -> None:
        """Save user edits. The untouched original stays in <id>.original.json."""
        self.report_path(mid).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    def _archive(self, mid: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for p in (self.report_path(mid), self.original_path(mid), self.llm_raw_path(mid)):
            if p.is_file():
                p.rename(p.with_name(f"{p.stem}.{stamp}{p.suffix}"))


# ---- Markdown export -----------------------------------------------------
def _ev_md(evs: list[Evidence]) -> list[str]:
    out = []
    for e in evs:
        out.append(f"  - [{format_ts(e.timestamp)}] **{e.speaker}** ({e.segment_id}): {e.segment_text}")
    return out


def report_to_markdown(report: MeetingReport, title: str = "Meeting Report") -> str:
    L: list[str] = [f"# {title}", "", "## A. Meeting Summary", report.summary or "_(none)_", "", "## B. Meeting Goals"]
    L += [f"- {g}" for g in report.goals] or ["_(none)_"]
    L += ["", "## C. Scope"] + ([f"- {s}" for s in report.scope] or ["_(none)_"])
    L += ["", "## D. Confirmed Decisions"]
    for d in report.decisions:
        L += [f"- **{d.id}** [{d.status.value}] {d.text}"] + _ev_md(d.evidence)
        L += [f"  - Review: {n}" for n in d.review_notes]
    if not report.decisions:
        L.append("_(none)_")
    L += ["", "## E. Proposed Ideas"]
    for i in report.proposed_ideas:
        L += [f"- **{i.id}** {i.text} (by {i.proposed_by or 'unknown'})"] + _ev_md(i.evidence)
    if not report.proposed_ideas:
        L.append("_(none)_")
    L += ["", "## F. Action Items by Person"]
    for person, items in report.tasks_by_person().items():
        L += ["", f"### {person}"]
        for a in items:
            L += [f"- **{a.id}** {a.description}",
                  f"  - Assigned by: {a.assigned_by or '—'} | Deadline: {a.deadline or '—'} | "
                  f"Status: {a.status.value} | Confidence: {a.confidence.value} | Time: {format_ts(a.timestamp)}"]
            L += _ev_md(a.evidence)
            L += [f"  - Review: {n}" for n in a.review_notes]
    if not report.action_items:
        L.append("_(none)_")
    L += ["", "## G. Dependencies"]
    L += [f"- {d.task_id} depends on {d.depends_on_task_id}: {d.description}" for d in report.dependencies] or ["_(none)_"]
    L += ["", "## H. Blockers"]
    for b in report.blockers:
        L += [f"- **{b.id}** {b.description} (raised by {b.raised_by or 'unknown'})"] + _ev_md(b.evidence)
    if not report.blockers:
        L.append("_(none)_")
    L += ["", "## I. Unresolved Questions"]
    for q in report.unresolved_questions:
        L += [f"- **{q.id}** {q.question} (asked by {q.raised_by or 'unknown'})"] + _ev_md(q.evidence)
    if not report.unresolved_questions:
        L.append("_(none)_")
    L += ["", "## J. Mentor Instructions"]
    for m in report.mentor_instructions:
        L += [f"- **{m.id}** {m.text} → {m.addressed_to or UNASSIGNED}"] + _ev_md(m.evidence)
    if not report.mentor_instructions:
        L.append("_(none)_")
    if report.conflicts:
        L += ["", "## Conflicting Instructions"]
        for c in report.conflicts:
            L += [f"- **{c.id}** {c.description}"] + _ev_md(c.evidence)
    if report.validator_notes or report.rejected_items:
        L += ["", "## Validator Notes"] + [f"- {n}" for n in report.validator_notes]
        L += [f"- Rejected ({r.section}): {r.text} — {r.reason}" for r in report.rejected_items]
    return "\n".join(L) + "\n"


def transcript_to_text(transcript: Transcript, speakers: SpeakerMap | None = None) -> str:
    lines = []
    for s in transcript.segments:
        who = speakers.display_name(s.speaker_id) if speakers else s.speaker_id
        lines.append(f"[{format_ts(s.start)}] {who}: {s.text}")
    return "\n".join(lines) + "\n"
