"""Text preprocessing — mirrors MAI_Group5/task1.py exactly.

The corpus-level filters from M1 (singleton removal, top-20 DF removal) are not
applied at inference time: they map onto the fitted TfidfVectorizer's
min_df / max_df, so they live inside models/text_pipeline.joblib once trained.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "stopwords_en.txt"


@lru_cache(maxsize=1)
def load_stopwords(path: str | Path | None = None) -> frozenset[str]:
    p = Path(path) if path is not None else STOPWORDS_PATH
    with p.open("r", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def tokenize(text: str | None) -> list[str]:
    """Tokenize one document the way M1 Task 1 did.

    Steps: regex `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`, lowercase, drop tokens with
    length < 2, drop stopwords. Returns [] for non-string input.
    """
    if not isinstance(text, str):
        return []
    stop = load_stopwords()
    return [w for w in TOKEN_RE.findall(text.lower()) if len(w) >= 2 and w not in stop]


def clean_text(text: str | None) -> str:
    """Return the space-joined cleaned token string used for model input."""
    return " ".join(tokenize(text))
