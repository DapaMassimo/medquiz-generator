from __future__ import annotations

from medquiz_generator.domain.text import excerpt_around, find_normalized

TEXT = (
    "La diagnosi richiede misurazioni ripetute con tecnica corretta. "
    "Il monitoraggio pressorio delle 24 ore (ABPM) è indicato in caso di sospetta "
    "ipertensione da camice bianco. La valutazione iniziale comprende esami di laboratorio."
)


def test_find_normalized_returns_span_in_original_text() -> None:
    span = find_normalized(TEXT, "monitoraggio  pressorio delle 24 ore, (abpm)")
    assert span is not None
    start, end = span
    # Trailing punctuation is not part of the normalised match.
    assert TEXT[start:end] == "monitoraggio pressorio delle 24 ore (ABPM"


def test_find_normalized_none_when_absent() -> None:
    assert find_normalized(TEXT, "frase inventata") is None
    assert find_normalized(TEXT, "") is None


def test_excerpt_around_contains_quote_with_bounded_margin() -> None:
    excerpt = excerpt_around(TEXT, "ipertensione da camice bianco", margin=20)
    assert "ipertensione da camice bianco" in excerpt
    assert len(excerpt) <= len("ipertensione da camice bianco") + 40


def test_excerpt_around_falls_back_to_head() -> None:
    assert excerpt_around(TEXT, "assente", fallback_chars=30) == TEXT[:30]
