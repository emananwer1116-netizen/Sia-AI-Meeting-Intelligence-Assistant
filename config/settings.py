"""Central configuration. All secrets come from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LITELLM_BASE_URL = "https://management.sprints.ai/litellm/"


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Secrets are excluded from repr so they can never leak into logs.
    elevenlabs_api_key: str = field(default="", repr=False)
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_stt_model: str = "scribe_v2"
    litellm_api_key: str = field(default="", repr=False)
    litellm_base_url: str = DEFAULT_LITELLM_BASE_URL
    litellm_model: str = ""
    litellm_json_mode: bool = False
    litellm_max_tokens: int | None = None
    litellm_timeout_secs: int = 300
    data_dir: Path = PROJECT_ROOT / "storage"
    max_audio_mb: int = 500
    max_audio_minutes: int = 240
    max_prompt_chars: int = 150_000

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sia.db"

    def ensure_dirs(self) -> None:
        for d in (self.recordings_dir, self.transcripts_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def missing_for_transcription(self) -> list[str]:
        return [] if self.elevenlabs_api_key.strip() else ["ELEVENLABS_API_KEY"]

    def missing_for_analysis(self) -> list[str]:
        missing = []
        if not self.litellm_api_key.strip():
            missing.append("LITELLM_API_KEY")
        if not self.litellm_base_url.strip():
            missing.append("LITELLM_BASE_URL")
        if not self.litellm_model.strip():
            missing.append("LITELLM_MODEL")
        return missing

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None, load_dotenv_file: bool = True) -> "Settings":
        """Build settings. Pass `env` (and load_dotenv_file=False) in tests."""
        if env is None:
            if load_dotenv_file:
                load_dotenv(PROJECT_ROOT / ".env", override=False)
            env = os.environ
        max_tokens = _to_int(env.get("LITELLM_MAX_TOKENS"), 0)
        return cls(
            elevenlabs_api_key=env.get("ELEVENLABS_API_KEY", "").strip(),
            elevenlabs_base_url=env.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").strip(),
            elevenlabs_stt_model=env.get("ELEVENLABS_STT_MODEL", "scribe_v2").strip() or "scribe_v2",
            litellm_api_key=env.get("LITELLM_API_KEY", "").strip(),
            litellm_base_url=(env.get("LITELLM_BASE_URL") or DEFAULT_LITELLM_BASE_URL).strip(),
            litellm_model=env.get("LITELLM_MODEL", "").strip(),
            litellm_json_mode=_to_bool(env.get("LITELLM_JSON_MODE"), False),
            litellm_max_tokens=max_tokens or None,
            litellm_timeout_secs=_to_int(env.get("LITELLM_TIMEOUT_SECS"), 300),
            data_dir=Path(env.get("SIA_DATA_DIR") or (PROJECT_ROOT / "storage")),
            max_audio_mb=_to_int(env.get("SIA_MAX_AUDIO_MB"), 500),
            max_audio_minutes=_to_int(env.get("SIA_MAX_AUDIO_MINUTES"), 240),
            max_prompt_chars=_to_int(env.get("SIA_MAX_PROMPT_CHARS"), 150_000),
        )
