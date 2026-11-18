"""Arabic/English-tolerant text matching helpers (used for evidence checks)."""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TABLE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})
_LATIN_ONLY = re.compile(r"^[a-z0-9 ]+$")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = _DIACRITICS.sub("", t).replace("\u0640", "")
    t = t.translate(_TABLE).casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_text(quote: str, text: str, min_len: int = 3) -> bool:
    nq, nt = normalize(quote), normalize(text)
    return len(nq) >= min_len and nq in nt


def name_in_text(name: str, text: str) -> bool:
    nn, nt = normalize(name), normalize(text)
    if len(nn) < 2 or not nt:
        return False
    if _LATIN_ONLY.match(nn):
        return re.search(rf"(?<![a-z0-9]){re.escape(nn)}(?![a-z0-9])", nt) is not None
    # Arabic names often carry clitics (يا أحمد / لأحمد): substring match.
    return nn in nt
