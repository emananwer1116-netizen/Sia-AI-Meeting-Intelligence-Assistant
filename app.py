"""Sia - AI Meeting Assistant (Streamlit UI). UI only: all logic lives in services/."""
from __future__ import annotations

import html
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import Settings
from database.db import Database
from schemas.meeting_report import UNASSIGNED, ItemStatus
from schemas.transcript import SpeakerInfo, SpeakerMap, SpeakerRole, format_ts, friendly_speaker_label
from services.audio_utils import ALLOWED_EXTENSIONS, AudioValidationError, inspect_audio
from services.jobs import JOBS
from services.litellm_client import LiteLLMClient, LLMError
from services.meeting_service import DeleteNotAllowed, delete_meeting
from services.report_service import AnalysisNotAllowed, ReportService, report_to_markdown, transcript_to_text
from services.transcription_service import TranscriptionError, TranscriptionService

SCREENS = ["1. Upload Meeting", "2. Transcript", "3. Speaker Identification",
           "4. Meeting Analysis", "5. Results", "6. Save / Open Meetings"]
PALETTE = ["#1D4ED8", "#0F766E", "#7C3AED", "#B45309", "#BE185D", "#475569"]
BADGES = {"Confirmed": ("#166534", "#DCFCE7"), "Needs Review": ("#92400E", "#FEF3C7"),
          "Unassigned": ("#374151", "#E5E7EB")}
NON_MENTOR_ROLES = [SpeakerRole.STUDENT, SpeakerRole.TEAM_LEAD, SpeakerRole.OTHER, SpeakerRole.UNKNOWN]

st.set_page_config(page_title="Sia - AI Meeting Assistant", page_icon="🗓️", layout="wide")

CSS = """
<style>
h1,h2,h3{color:#1E3A8A}
.sia-title{font-size:2rem;font-weight:700;color:#1D4ED8;margin:0}
.sia-sub{color:#64748B;margin:0 0 .6rem 0}
.seg{border-left:4px solid #1D4ED8;background:#F8FAFC;padding:.45rem .8rem;margin:.35rem 0;border-radius:6px}
.seg .meta{font-size:.8rem;color:#64748B}
.seg .txt{font-size:1rem;line-height:1.7;unicode-bidi:plaintext}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.75rem;font-weight:600;margin-right:6px}
.ev{border-left:3px solid #CBD5E1;padding:.2rem .6rem;margin:.25rem 0;color:#334155;font-size:.92rem;unicode-bidi:plaintext}
.note{color:#92400E;font-size:.85rem}
.item{unicode-bidi:plaintext;font-size:1rem;line-height:1.6}
</style>
"""


@dataclass
class Ctx:
    settings: Settings
    db: Database
    tx: TranscriptionService
    rep: ReportService


@st.cache_resource
def get_ctx() -> Ctx:
    settings = Settings.load()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init_schema()
    return Ctx(settings, db, TranscriptionService(settings, db), ReportService(settings, db))


# ---------------------------------------------------------------- helpers
def esc(text: str | None) -> str:
    return html.escape(text or "")


def badge(status: str) -> str:
    fg, bg = BADGES.get(status, ("#374151", "#E5E7EB"))
    return f'<span class="badge" style="color:{fg};background:{bg}">{esc(status)}</span>'


def fmt_duration(secs: float | None) -> str:
    return "unknown" if secs is None else f"{int(secs // 60)} min {int(secs % 60)} s"


def goto(screen: str | None = None, meeting_id: str | None = None) -> None:
    if screen:
        st.session_state["_pending_screen"] = screen
    if meeting_id:
        st.session_state["_pending_meeting"] = meeting_id
    st.rerun()


def evidence_html(evs) -> str:
    return "".join(
        f'<div class="ev">[{format_ts(e.timestamp)}] <b>{esc(e.speaker)}</b> · {esc(e.segment_id)}<br>'
        f'<span dir="auto">{esc(e.segment_text)}</span></div>' for e in evs)


def item_html(title: str, status: str | None, notes: list[str], evs, extra: str = "") -> str:
    head = f'<div class="item" dir="auto">{badge(status) if status else ""}{esc(title)}</div>'
    meta = f'<div style="color:#64748B;font-size:.85rem">{extra}</div>' if extra else ""
    nts = "".join(f'<div class="note">⚠ {esc(n)}</div>' for n in notes)
    return head + meta + nts + evidence_html(evs)


def need_meeting(m: dict | None) -> bool:
    if m is None:
        st.info("No meeting selected. Upload a recording in screen 1.")
        return False
    return True


def need_transcript(ctx: Ctx, m: dict):
    t = ctx.tx.load_transcript(m["id"])
    if t is None:
        st.info("This meeting has no transcript yet. Transcribe it in screen 1.")
    return t


# ---------------------------------------------------------------- screen 1
def screen_upload(ctx: Ctx, m: dict | None) -> None:
    st.header("Upload Meeting")
    st.caption("Uploading only saves the recording locally. Nothing is sent to ElevenLabs until you confirm below.")
    exts = [e.lstrip(".") for e in sorted(ALLOWED_EXTENSIONS)]
    uploaded = st.file_uploader("Meeting recording", type=exts, key="uploader")
    if uploaded is not None:
        data = uploaded.getvalue()
        try:
            info = inspect_audio(uploaded.name, data, ctx.settings)
        except AudioValidationError as exc:
            st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("File", info.filename)
            c2.metric("Size", f"{info.size_mb:.1f} MB")
            c3.metric("Duration", fmt_duration(info.duration_secs))
            if info.duration_secs is None:
                st.warning("Duration could not be read from this file; ElevenLabs will bill by actual audio length.")
            if info.size_mb <= 30:
                st.audio(data)
            dup = ctx.db.find_by_sha256(info.sha256)
            if dup:
                if m and dup["id"] == m["id"]:
                    st.success("Recording saved. Continue below.")
                else:
                    st.warning(f"This exact recording is already saved as “{dup['title']}”. "
                               "It was not added again, so it cannot be sent to ElevenLabs twice by accident.")
                    if st.button("Open that meeting", key="open_dup"):
                        goto("1. Upload Meeting", dup["id"])
            else:
                title = st.text_input("Meeting title", value=Path(info.filename).stem, key="new_title")
                if st.button("Save recording", type="primary", key="save_rec"):
                    target = ctx.settings.recordings_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
                    target.mkdir(parents=True, exist_ok=True)
                    path = target / info.safe_filename
                    path.write_bytes(data)
                    mid = ctx.db.create_meeting(title=title.strip() or info.filename, audio_filename=info.filename,
                                                audio_path=str(path), audio_sha256=info.sha256,
                                                audio_size_bytes=info.size_bytes, audio_duration_secs=info.duration_secs)
                    goto("1. Upload Meeting", mid)
    if m is None:
        return
    st.divider()
    transcription_panel(ctx, m)


def transcription_panel(ctx: Ctx, m: dict) -> None:
    mid, status, key = m["id"], m["transcription_status"], f"tx:{m['id']}"
    st.subheader(f"Transcription · {m['title']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recording", m["audio_filename"])
    c2.metric("Duration", fmt_duration(m["audio_duration_secs"]))
    c3.metric("Sent to ElevenLabs", f"{m['transcription_attempts']} time(s)")
    missing = ctx.settings.missing_for_transcription()
    if missing and status != "done":
        st.error("Missing in .env: " + ", ".join(missing) + ". See README.")
        return
    if status == "in_progress":
        if JOBS.is_running(key):
            st.info("Transcribing with ElevenLabs… keep this page open. It refreshes automatically.")
            time.sleep(2)
            st.rerun()
        else:
            st.warning("This request looks interrupted (the app was closed or restarted while it ran). "
                       "Check your ElevenLabs usage before sending again.")
            if st.button("Mark as failed (I checked my usage)", key=f"reset_{mid}"):
                ctx.tx.reset_interrupted(mid)
                st.rerun()
        return
    if status == "done":
        st.success("Transcript is saved locally. Re-opening this meeting never uses ElevenLabs again.")
        if st.button("Open transcript →", key=f"open_tx_{mid}"):
            goto("2. Transcript")
        send_form(ctx, m, resend=True)
        return
    if status == "failed":
        st.error(m["transcription_error"] or "The previous request failed.")
        if ctx.tx.raw_path(mid).is_file() and st.button("Rebuild from saved response (free)", key=f"rb_{mid}"):
            try:
                ctx.tx.rebuild_from_raw(mid)
            except TranscriptionError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    send_form(ctx, m, resend=(status == "failed"))


def send_form(ctx: Ctx, m: dict, resend: bool) -> None:
    mid, dur = m["id"], m["audio_duration_secs"]
    box = st.expander("Send this recording to ElevenLabs again (uses credits)") if m["transcription_status"] == "done" \
        else st.container()
    with box:
        st.warning(f"This request will consume ElevenLabs credits for about {fmt_duration(dur)} of audio. "
                   "Free-tier credits are limited.")
        lang = st.selectbox("Language", ["Auto-detect (recommended for mixed Arabic + English)", "ar", "en"],
                            key=f"lang_{mid}")
        n_spk = st.number_input("Max number of speakers (0 = let ElevenLabs decide)", 0, 32, 0, key=f"nspk_{mid}")
        ok1 = st.checkbox("I understand this request will use my ElevenLabs credits.", key=f"ok1_{mid}")
        ok2 = True
        if resend:
            ok2 = st.checkbox("This recording was already sent before (it may have been billed). Send it again.",
                              key=f"ok2_{mid}")
        if st.button("Start transcription" if not resend else "Send again", type="primary",
                     disabled=not (ok1 and ok2), key=f"start_{mid}"):
            try:
                ticket = ctx.tx.prepare(mid, confirmed=ok1, allow_resend=resend and ok2,
                                        language_code=None if lang.startswith("Auto") else lang,
                                        num_speakers=int(n_spk) or None)
            except TranscriptionError as exc:
                st.error(str(exc))
            else:
                JOBS.start(f"tx:{mid}", lambda: ctx.tx.execute(ticket))
                st.rerun()


# ---------------------------------------------------------------- screen 2
def screen_transcript(ctx: Ctx, m: dict | None) -> None:
    st.header("Transcript")
    if not need_meeting(m):
        return
    t = need_transcript(ctx, m)
    if t is None:
        return
    speakers = ctx.rep.load_speakers(m["id"])
    ids = t.speaker_ids()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Language", t.language_code or "?")
    c2.metric("Duration", fmt_duration(t.audio_duration_secs))
    c3.metric("Segments", len(t.segments))
    c4.metric("Speakers", len(ids))
    st.caption("Read-only view. The transcript is never modified automatically.")
    query = st.text_input("Filter (view only)", key=f"flt_{m['id']}").strip().casefold()
    parts = []
    for seg in t.segments:
        if query and query not in seg.text.casefold():
            continue
        color = PALETTE[ids.index(seg.speaker_id) % len(PALETTE)]
        who = speakers.display_name(seg.speaker_id) if speakers else friendly_speaker_label(seg.speaker_id)
        parts.append(
            f'<div class="seg" style="border-left-color:{color}"><div class="meta"><b style="color:{color}">'
            f'{esc(who)}</b> · {format_ts(seg.start)} · {esc(seg.id)}</div>'
            f'<div class="txt" dir="auto">{esc(seg.text)}</div></div>')
    st.markdown("".join(parts) or "_No matching segments._", unsafe_allow_html=True)
    st.download_button("Download transcript (.txt)", transcript_to_text(t, speakers),
                       file_name=f"{m['title']}_transcript.txt", key=f"dl_tx_{m['id']}")


# ---------------------------------------------------------------- screen 3
def screen_speakers(ctx: Ctx, m: dict | None) -> None:
    st.header("Speaker Identification")
    if not need_meeting(m):
        return
    t = need_transcript(ctx, m)
    if t is None:
        return
    mid, ids = m["id"], t.speaker_ids()
    saved = ctx.rep.load_speakers(mid) or SpeakerMap()
    st.caption("Roles are saved for this meeting only. Sia never guesses who the Mentor is.")
    NONE = "— not selected —"
    with st.form(f"speakers_{mid}"):
        entries: dict[str, dict] = {}
        for sid in ids:
            cur = saved.get(sid)
            lines = [s.text for s in t.segments if s.speaker_id == sid][:2]
            st.markdown(f"**{friendly_speaker_label(sid)}**")
            for ln in lines:
                st.markdown(f'<div class="ev" dir="auto">{esc(ln[:160])}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 2, 2])
            name = c1.text_input("Name", value=cur.name if cur else "", key=f"n_{mid}_{sid}")
            aliases = c2.text_input("Other spellings (optional, comma separated)",
                                    value=", ".join(cur.aliases) if cur else "", key=f"a_{mid}_{sid}",
                                    help="e.g. the Arabic spelling of an English name, so Sia can verify the name was said.")
            role_default = cur.role if cur and cur.role != SpeakerRole.MENTOR else SpeakerRole.UNKNOWN
            role = c3.selectbox("Role", [r.value for r in NON_MENTOR_ROLES], index=[r.value for r in NON_MENTOR_ROLES].index(role_default.value),
                                key=f"r_{mid}_{sid}")
            entries[sid] = {"name": name, "aliases": aliases, "role": role}
            st.divider()
        mentor_default = saved.mentor.speaker_id if saved.mentor else NONE
        mentor = st.radio("Who is the Mentor?", [NONE, *ids], index=([NONE, *ids].index(mentor_default)
                          if mentor_default in ids or mentor_default == NONE else 0),
                          format_func=lambda s: s if s == NONE else friendly_speaker_label(s), key=f"mentor_{mid}")
        submitted = st.form_submit_button("Save speakers for this meeting", type="primary")
    if submitted:
        infos = []
        for sid in ids:
            e = entries[sid]
            role = SpeakerRole.MENTOR if sid == mentor else SpeakerRole(e["role"])
            infos.append(SpeakerInfo(speaker_id=sid, name=e["name"].strip(), role=role,
                                     aliases=[a.strip() for a in e["aliases"].split(",") if a.strip()]))
        problems = ctx.rep.save_speakers(mid, SpeakerMap(speakers=infos), t)
        if problems:
            st.error(" ".join(problems))
        else:
            st.success("Speakers saved. You can now analyze the meeting.")
            if m["analysis_status"] == "done":
                st.info("A report already exists from the previous speaker setup. Re-run the analysis to update it.")


# ---------------------------------------------------------------- screen 4
def screen_analysis(ctx: Ctx, m: dict | None) -> None:
    st.header("Meeting Analysis")
    if not need_meeting(m):
        return
    mid, key = m["id"], f"an:{m['id']}"
    t = ctx.tx.load_transcript(mid)
    speakers = ctx.rep.load_speakers(mid)
    problems = speakers.problems(t.speaker_ids()) if (speakers and t) else ["Speakers not saved."]
    missing = ctx.settings.missing_for_analysis()
    st.markdown(f"- Transcript: {'✅' if t else '❌ missing'}\n"
                f"- Speakers & Mentor: {'✅ ' + speakers.mentor.display_name if speakers and not problems else '❌ ' + ' '.join(problems)}\n"
                f"- LiteLLM settings: {'✅ model ' + ctx.settings.litellm_model if not missing else '❌ missing ' + ', '.join(missing)}")
    with st.expander("LiteLLM connection helper"):
        st.caption(f"Base URL: {ctx.settings.litellm_base_url}")
        if st.button("List available models", key="list_models"):
            try:
                client = LiteLLMClient(ctx.settings.litellm_base_url, ctx.settings.litellm_api_key, ctx.settings.litellm_model,
                                       timeout=60)
                st.code("\n".join(client.list_models()) or "(no models returned)")
            except LLMError as exc:
                st.error(str(exc))
    status = m["analysis_status"]
    running = JOBS.is_running(key)
    if status == "in_progress" and running:
        st.info("Analyzing the meeting… this can take a minute. The page refreshes automatically.")
        time.sleep(2)
        st.rerun()
    if status == "in_progress" and not running:
        st.warning("An analysis looks interrupted.")
        if st.button("Reset analysis state", key=f"ra_{mid}"):
            ctx.rep.reset_interrupted_analysis(mid)
            st.rerun()
        return
    if status == "failed":
        st.error(m["analysis_error"] or "The last analysis failed.")
    ready = t is not None and not problems and not missing
    again = status == "done"
    ok = True
    if again:
        st.success("This meeting has been analyzed.")
        if st.button("Open results →", key=f"to_res_{mid}"):
            goto("5. Results")
        ok = st.checkbox("Run the analysis again (new LLM request; the old report is archived).", key=f"again_{mid}")
    if st.button("Analyze Meeting", type="primary", disabled=not (ready and ok), key=f"analyze_{mid}"):
        try:
            sp = ctx.rep.prepare_analysis(mid, t, allow_reanalyze=again and ok)
        except AnalysisNotAllowed as exc:
            st.error(str(exc))
        else:
            JOBS.start(key, lambda: ctx.rep.execute_analysis(mid, t, sp))
            st.rerun()


# ---------------------------------------------------------------- screen 5
def screen_results(ctx: Ctx, m: dict | None) -> None:
    st.header("Results")
    if not need_meeting(m):
        return
    mid = m["id"]
    rep = ctx.rep.load_report(mid)
    if rep is None:
        st.info("No report yet. Run the analysis in screen 4.")
        return
    work = rep.model_copy(deep=True)
    people = [s.display_name for s in (ctx.rep.load_speakers(mid) or SpeakerMap()).speakers]
    tabs = st.tabs(["Summary", "Decisions", "Proposed Ideas", "Action Items", "By Person", "Dependencies",
                    "Blockers", "Open Questions", "Mentor Instructions", "Validator"])
    with tabs[0]:
        work.summary = st.text_area("Meeting Summary", rep.summary, height=140, key=f"sum_{mid}")
        work.goals = [g.strip() for g in st.text_area("Meeting Goals (one per line)", "\n".join(rep.goals),
                                                      key=f"goals_{mid}").splitlines() if g.strip()]
        work.scope = [g.strip() for g in st.text_area("Scope (one per line)", "\n".join(rep.scope),
                                                      key=f"scope_{mid}").splitlines() if g.strip()]
    with tabs[1]:
        if not rep.decisions:
            st.caption("No confirmed decisions.")
        else:
            df = pd.DataFrame([{"ID": d.id, "Decision": d.text, "Status": d.status.value, "Confidence": d.confidence.value}
                               for d in rep.decisions])
            edited = st.data_editor(df, key=f"dec_{mid}", hide_index=True, use_container_width=True,
                                    disabled=["ID", "Confidence"], column_config={"Status": st.column_config.SelectboxColumn(
                                        options=["Confirmed", "Needs Review"])})
            for d, (_, row) in zip(work.decisions, edited.iterrows()):
                d.text, d.status = str(row["Decision"]), ItemStatus(row["Status"])
            for d in rep.decisions:
                with st.expander(f"{d.id} · evidence"):
                    st.markdown(item_html(d.text, d.status.value, d.review_notes, d.evidence), unsafe_allow_html=True)
    with tabs[2]:
        for i in rep.proposed_ideas:
            st.markdown(item_html(i.text, None, i.review_notes, i.evidence, f"Proposed by {esc(i.proposed_by or 'unknown')}"),
                        unsafe_allow_html=True)
        if not rep.proposed_ideas:
            st.caption("No proposed ideas.")
    with tabs[3]:
        if not rep.action_items:
            st.caption("No action items.")
        else:
            options = list(dict.fromkeys([*people, *[a.assigned_to for a in rep.action_items], UNASSIGNED]))
            df = pd.DataFrame([{"ID": a.id, "Task": a.description, "Assigned To": a.assigned_to, "Deadline": a.deadline or "",
                                "Status": a.status.value, "Reviewed": a.user_reviewed, "Assigned By": a.assigned_by or "",
                                "Confidence": a.confidence.value, "Time": format_ts(a.timestamp)} for a in rep.action_items])
            edited = st.data_editor(df, key=f"act_{mid}", hide_index=True, use_container_width=True,
                                    disabled=["ID", "Assigned By", "Confidence", "Time"],
                                    column_config={"Assigned To": st.column_config.SelectboxColumn(options=options),
                                                   "Status": st.column_config.SelectboxColumn(options=[s.value for s in ItemStatus])})
            for a, (_, row) in zip(work.action_items, edited.iterrows()):
                a.description, a.assigned_to = str(row["Task"]), str(row["Assigned To"])
                a.deadline = str(row["Deadline"]).strip() or None
                a.status, a.user_reviewed = ItemStatus(row["Status"]), bool(row["Reviewed"])
                if a.assigned_to == UNASSIGNED:
                    a.status = ItemStatus.UNASSIGNED
                elif a.status == ItemStatus.UNASSIGNED:
                    a.status = ItemStatus.NEEDS_REVIEW
            st.caption("Edit task, owner, deadline or status directly, tick “Reviewed”, then press Save changes below.")
            for a in rep.action_items:
                with st.expander(f"{a.id} · {a.assigned_to} · {a.description[:70]}"):
                    st.markdown(item_html(a.description, a.status.value, a.review_notes, a.evidence,
                                          f"Assigned by: {esc(a.assigned_by or '—')} · Deadline: {esc(a.deadline or '—')} · "
                                          f"Confidence: {a.confidence.value}"), unsafe_allow_html=True)
    with tabs[4]:
        groups = work.tasks_by_person()
        choice = st.selectbox("Person", ["All", *groups], key=f"person_{mid}")
        for person, items in groups.items():
            if choice not in ("All", person):
                continue
            st.subheader(f"{person} ({len(items)})")
            for a in items:
                with st.container(border=True):
                    st.markdown(item_html(a.description, a.status.value, a.review_notes, a.evidence[:1],
                                          f"Deadline: {esc(a.deadline or '—')} · by {esc(a.assigned_by or 'self')}"),
                                unsafe_allow_html=True)
    with tabs[5]:
        for d in rep.dependencies:
            st.markdown(item_html(f"{d.task_id} depends on {d.depends_on_task_id}: {d.description}", None, d.review_notes,
                                  d.evidence), unsafe_allow_html=True)
        if not rep.dependencies:
            st.caption("No dependencies.")
    with tabs[6]:
        for b in rep.blockers:
            st.markdown(item_html(b.description, None, b.review_notes, b.evidence, f"Raised by {esc(b.raised_by or 'unknown')}"),
                        unsafe_allow_html=True)
        if not rep.blockers:
            st.caption("No blockers.")
    with tabs[7]:
        for q in rep.unresolved_questions:
            st.markdown(item_html(q.question, None, q.review_notes, q.evidence, f"Asked by {esc(q.raised_by or 'unknown')}"),
                        unsafe_allow_html=True)
        if not rep.unresolved_questions:
            st.caption("No unresolved questions.")
    with tabs[8]:
        for mi in rep.mentor_instructions:
            st.markdown(item_html(mi.text, None, mi.review_notes, mi.evidence, f"To: {esc(mi.addressed_to or UNASSIGNED)}"),
                        unsafe_allow_html=True)
        if rep.conflicts:
            st.subheader("Conflicting instructions")
            for c in rep.conflicts:
                st.markdown(item_html(c.description, "Needs Review", c.review_notes, c.evidence), unsafe_allow_html=True)
        if not rep.mentor_instructions:
            st.caption("No Mentor instructions.")
    with tabs[9]:
        for n in rep.validator_notes:
            st.warning(n)
        if rep.rejected_items:
            st.caption("Items proposed by the model that Sia refused because they could not be tied to the transcript:")
            st.dataframe(pd.DataFrame([r.model_dump() for r in rep.rejected_items]), hide_index=True, use_container_width=True)
        else:
            st.success("Nothing was rejected.")
    st.divider()
    c1, c2, c3 = st.columns(3)
    if c1.button("Save changes", type="primary", key=f"save_rep_{mid}"):
        ctx.rep.save_report(mid, work)
        st.success("Saved. The untouched original is kept separately.")
    c2.download_button("Download report (.md)", report_to_markdown(work, m["title"]),
                       file_name=f"{m['title']}_report.md", key=f"dl_md_{mid}")
    c3.download_button("Download report (.json)", work.model_dump_json(indent=2),
                       file_name=f"{m['title']}_report.json", key=f"dl_js_{mid}")


# ---------------------------------------------------------------- screen 6
def screen_library(ctx: Ctx, m: dict | None) -> None:
    st.header("Save / Open Meetings")
    st.caption(f"Everything is stored locally in: {ctx.settings.data_dir}. Opening a saved meeting never calls ElevenLabs.")
    meetings = ctx.db.list_meetings()
    if not meetings:
        st.info("No saved meetings yet.")
        return
    for mt in meetings:
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 3, 1])
            c1.markdown(f"**{esc(mt['title'])}**  \n{esc(mt['audio_filename'])} · {esc(mt['created_at'][:16].replace('T', ' '))}")
            c2.markdown(f"Transcript: `{mt['transcription_status']}`  \nAnalysis: `{mt['analysis_status']}`")
            if c3.button("Open", key=f"open_{mt['id']}"):
                goto("5. Results" if mt["analysis_status"] == "done" else "1. Upload Meeting", mt["id"])
            with st.expander("Delete this meeting"):
                if mt["transcription_status"] == "done":
                    st.warning("A finished transcript exists. Deleting it means transcribing again will use ElevenLabs "
                               "credits again. Download what you need first.")
                st.caption("Permanently deletes the recording, transcript, report and the meeting record.")
                sure = st.checkbox("I understand this cannot be undone.", key=f"del_ok_{mt['id']}")
                if st.button("Delete meeting", disabled=not sure, key=f"del_{mt['id']}"):
                    try:
                        delete_meeting(ctx.settings, ctx.db, mt["id"])
                    except DeleteNotAllowed as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
    if m:
        st.subheader(f"Export · {m['title']}")
        c1, c2, c3 = st.columns(3)
        rep = ctx.rep.load_report(m["id"])
        t = ctx.tx.load_transcript(m["id"])
        if rep:
            c1.download_button("Report (.md)", report_to_markdown(rep, m["title"]), file_name=f"{m['title']}_report.md",
                               key=f"lib_md_{m['id']}")
        if t:
            c2.download_button("Transcript (.txt)", transcript_to_text(t, ctx.rep.load_speakers(m["id"])),
                               file_name=f"{m['title']}_transcript.txt", key=f"lib_tx_{m['id']}")
        audio = Path(m["audio_path"])
        if audio.is_file() and audio.stat().st_size < 200 * 1024 * 1024:
            c3.download_button("Original recording", audio.read_bytes(), file_name=m["audio_filename"], key=f"lib_au_{m['id']}")


# ---------------------------------------------------------------- main
def main() -> None:
    ctx = get_ctx()
    st.markdown(CSS, unsafe_allow_html=True)
    if "_pending_screen" in st.session_state:
        st.session_state["screen"] = st.session_state.pop("_pending_screen")
    if "_pending_meeting" in st.session_state:
        st.session_state["meeting_id"] = st.session_state.pop("_pending_meeting")
    st.session_state.setdefault("screen", SCREENS[0])
    meetings = ctx.db.list_meetings()
    ids = [x["id"] for x in meetings]
    if st.session_state.get("meeting_id") not in ids:
        st.session_state["meeting_id"] = ids[0] if ids else None
    with st.sidebar:
        st.markdown('<p class="sia-title">Sia</p><p class="sia-sub">AI Meeting Assistant</p>', unsafe_allow_html=True)
        st.radio("Screens", SCREENS, key="screen", label_visibility="collapsed")
        if ids:
            titles = {x["id"]: x["title"] for x in meetings}
            st.selectbox("Meeting", ids, key="meeting_id", format_func=lambda i: titles.get(i, i))
    m = ctx.db.get_meeting(st.session_state["meeting_id"]) if st.session_state.get("meeting_id") else None
    {SCREENS[0]: screen_upload, SCREENS[1]: screen_transcript, SCREENS[2]: screen_speakers,
     SCREENS[3]: screen_analysis, SCREENS[4]: screen_results, SCREENS[5]: screen_library}[st.session_state["screen"]](ctx, m)


main()
