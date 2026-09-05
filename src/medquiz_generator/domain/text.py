"""Pure text helpers shared by the deterministic checks and the planner.

Everything here is framework-free and deterministic so it can be unit-tested exhaustively
and reused by adapters (e.g. to locate a quote in a page) without duplicating logic.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    " ": " ",
    "­": "",
}


def normalize_for_match(text: str) -> str:
    """Lower-case, unify quotes/dashes, drop punctuation and collapse whitespace.

    Used to check that a quote produced by the generator is *really* present in the source
    section: small differences in quotes, dashes, hyphenation and spacing must not break the
    match, while any change of wording must.
    """
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Re-join words hyphenated across line breaks ("infarto mio-\ncardico").
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def contains_normalized(haystack: str, needle: str) -> bool:
    """True when ``needle`` occurs in ``haystack`` after normalisation of both sides."""
    n = normalize_for_match(needle)
    if not n:
        return False
    return n in normalize_for_match(haystack)


def _prepare(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def find_normalized(haystack: str, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``haystack`` under the same normalisation as
    :func:`contains_normalized`; returns character offsets into the (prepared) haystack."""
    target = normalize_for_match(needle)
    if not target:
        return None
    prepared = _prepare(haystack)
    normalized_chars: list[str] = []
    origins: list[int] = []
    pending_space = False
    for index, char in enumerate(prepared):
        lowered = char.lower()
        if lowered.isalnum() or lowered == "_":
            if pending_space and normalized_chars:
                normalized_chars.append(" ")
                origins.append(index)
            pending_space = False
            normalized_chars.append(lowered)
            origins.append(index)
        else:
            pending_space = True
    normalized = "".join(normalized_chars)
    position = normalized.find(target)
    if position < 0:
        return None
    return origins[position], origins[position + len(target) - 1] + 1


def excerpt_around(text: str, quote: str, margin: int = 400, fallback_chars: int = 1500) -> str:
    """The quote with ``margin`` characters of context on each side (or the section head
    when the quote cannot be located). Used to give the judge a bounded passage."""
    span = find_normalized(text, quote)
    prepared = _prepare(text)
    if span is None:
        return prepared[:fallback_chars]
    start, end = span
    return prepared[max(0, start - margin) : min(len(prepared), end + margin)]


def estimate_tokens(text: str) -> int:
    """Cheap, provider-independent token estimate (~4 chars per token) for budgeting."""
    return max(1, len(text) // 4)


def word_count(text: str) -> int:
    return len(text.split())
