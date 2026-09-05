from __future__ import annotations

from medquiz_generator.domain.text import contains_normalized, estimate_tokens, normalize_for_match


def test_normalize_collapses_whitespace_and_punctuation() -> None:
    assert normalize_for_match("  Ciao,   mondo!  ") == "ciao mondo"


def test_normalize_unifies_quotes_and_dashes() -> None:
    assert normalize_for_match("l’aspetto “a becco d’uccello” – tipico") == normalize_for_match(
        "l'aspetto \"a becco d'uccello\" - tipico"
    )


def test_normalize_rejoins_hyphenated_line_breaks() -> None:
    assert normalize_for_match("infarto mio-\ncardico") == "infarto miocardico"


def test_contains_normalized_true_with_layout_noise() -> None:
    haystack = "La diagnosi richiede   misurazioni ripetute,\ncon tecnica corretta."
    assert contains_normalized(haystack, "misurazioni ripetute con tecnica corretta")


def test_contains_normalized_false_when_wording_changes() -> None:
    haystack = "La diagnosi richiede misurazioni ripetute."
    assert not contains_normalized(haystack, "La diagnosi richiede misurazioni singole")


def test_contains_normalized_empty_needle_is_false() -> None:
    assert not contains_normalized("qualcosa", "   ,,, ")


def test_estimate_tokens_is_positive() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100
