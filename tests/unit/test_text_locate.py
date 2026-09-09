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


PARAGRAPHS = "\n\n".join(
    [
        "Introduzione al capitolo.",
        "Figura 25.2 Attività motoria normale dell'esofago.",
        "La manometria misura la pressione.",
        "L'esame permette di valutare: la peristalsi e il tono dello sfintere.",
        "Si effettua con una sonda naso-esofagea.",
        "Figura 25.3 Classificazione dei disordini.",
        "EGJ = giunzione esofago-gastrica.",
    ]
)


def test_paragraphs_around_returns_whole_paragraphs_with_the_quote_span() -> None:
    from medquiz_generator.domain.text import paragraphs_around

    paras, before, after = paragraphs_around(PARAGRAPHS, "permette di valutare", context=1)
    assert [p.text[:12] for p in paras] == ["La manometri", "L'esame perm", "Si effettua "]
    assert before and after
    quoted = paras[1]
    assert quoted.quote_span is not None
    start, end = quoted.quote_span
    assert quoted.text[start:end] == "permette di valutare"
    assert paras[0].quote_span is None and paras[2].quote_span is None


def test_paragraphs_around_spans_two_paragraphs_and_falls_back_to_head() -> None:
    from medquiz_generator.domain.text import paragraphs_around

    paras, _, _ = paragraphs_around(PARAGRAPHS, "dello sfintere. Si effettua", context=0)
    assert [p.quote_span is not None for p in paras] == [True, True]
    head, before, after = paragraphs_around(PARAGRAPHS, "frase assente", fallback_paragraphs=2)
    assert len(head) == 2 and not before and after
