"""ElevenLabs Speech-to-Text client.

Verified against the official docs (POST /v1/speech-to-text, multipart/form-data,
`xi-api-key` header, fields model_id/file/diarize/timestamps_granularity/
language_code/num_speakers/tag_audio_events). Response: `words[]` with
`speaker_id`, `start`, `end`, `type` ('word' | 'spacing' | 'audio_event').

By design there is NO automatic retry: every request can consume credits.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import requests


class ElevenLabsError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ElevenLabsClient:
    def __init__(self, api_key: str, base_url: str = "https://api.elevenlabs.io",
                 model_id: str = "scribe_v2", timeout: tuple[int, int] = (30, 1800),
                 session: Any | None = None):
        if not api_key:
            raise ElevenLabsError("ELEVENLABS_API_KEY is not set.")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self._session = session or requests.Session()  # no retry adapters mounted

    def transcribe(self, audio_path: Path | str, *, language_code: str | None = None,
                   num_speakers: int | None = None, tag_audio_events: bool = False) -> dict:
        path = Path(audio_path)
        if not path.is_file():
            raise ElevenLabsError(f"Recording file not found: {path.name}")
        data: dict[str, str] = {
            "model_id": self.model_id,
            "diarize": "true",
            "timestamps_granularity": "word",
            "tag_audio_events": "true" if tag_audio_events else "false",
        }
        if language_code:
            data["language_code"] = language_code
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            with path.open("rb") as fh:
                resp = self._session.post(
                    f"{self.base_url}/v1/speech-to-text",
                    headers={"xi-api-key": self._api_key},
                    data=data,
                    files={"file": (path.name, fh, mime)},
                    timeout=self.timeout,
                )
        except requests.Timeout as exc:
            raise ElevenLabsError(
                "The request timed out. ElevenLabs may still have processed (and billed) it; "
                "check your usage before sending again."
            ) from exc
        except requests.ConnectionError as exc:
            raise ElevenLabsError(
                "Could not connect to ElevenLabs. Check your internet connection."
            ) from exc
        except requests.RequestException as exc:
            raise ElevenLabsError(f"Request failed: {type(exc).__name__}") from exc
        return self._handle(resp)

    # ------------------------------------------------------------------
    def _handle(self, resp: Any) -> dict:
        status = resp.status_code
        if status == 200:
            try:
                body = resp.json()
            except ValueError as exc:
                raise ElevenLabsError("ElevenLabs returned a non-JSON response.", status) from exc
            if "transcripts" in body:
                raise ElevenLabsError("Unexpected multichannel response.", status)
            if "words" not in body and "text" not in body:
                raise ElevenLabsError("Unexpected response shape (no 'words'/'text').", status)
            return body
        if status == 202:
            raise ElevenLabsError("ElevenLabs accepted the job asynchronously (webhook mode).", status)
        detail = self._detail(resp)
        messages = {
            401: "Invalid or missing ElevenLabs API key (401).",
            403: "Access denied by ElevenLabs (403). Check your plan/permissions.",
            413: "File is too large for ElevenLabs (413).",
            422: "ElevenLabs rejected the request parameters (422).",
            429: "ElevenLabs rate limit or concurrency limit reached (429). Wait and try again manually.",
        }
        if status == 402 or "quota" in detail.lower():
            base = "ElevenLabs credits/quota exhausted or payment required."
        elif status in messages:
            base = messages[status]
        elif status >= 500:
            base = f"ElevenLabs server error ({status}). The request may or may not have been billed."
        else:
            base = f"ElevenLabs returned HTTP {status}."
        raise ElevenLabsError(f"{base} {detail}".strip(), status)

    @staticmethod
    def _detail(resp: Any) -> str:
        try:
            body = resp.json()
        except ValueError:
            return (getattr(resp, "text", "") or "")[:300]
        detail = body.get("detail", body) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("status") or detail)[:300]
        if isinstance(detail, list):
            return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)[:300]
        return str(detail)[:300]
