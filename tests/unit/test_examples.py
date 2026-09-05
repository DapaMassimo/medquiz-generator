from __future__ import annotations

import logging
from pathlib import Path

import pytest

from medquiz_generator.application.examples import (
    format_examples,
    load_ssm_examples,
    parse_ssm_examples,
)
from tests.conftest import EXAMPLES_PATH

SAMPLE = """SSM 2026 quesiti: le prime 5 domande commentate
Domanda 1

Un uomo di 50 anni si presenta con disfagia.
Quale diagnosi?

    ☑ A: Acalasia esofagea (risposta corretta)
    ☐ B: Stenosi organica
    ☐ C: Malformazione
    ☐ D: Sindrome di Mallory-Weiss
    ☐ E: Neoplasia

Commento: La diagnosi di acalasia si basa sul quadro clinico.
Seconda riga del commento.
Domanda 2
Domanda senza marcatori.
A) prima
B) seconda (corretta)
C) terza
D) quarta
E) quinta
Domanda 3
Blocco rotto senza opzioni.
"""


def test_parse_examples_reads_question_options_correct_and_commentary() -> None:
    examples = parse_ssm_examples(SAMPLE)
    assert [e.number for e in examples] == [1, 2]
    first = examples[0]
    assert first.question == "Un uomo di 50 anni si presenta con disfagia. Quale diagnosi?"
    assert [o.id for o in first.options] == ["A", "B", "C", "D", "E"]
    assert first.options[0].text == "Acalasia esofagea"
    assert first.correct_option_id == "A"
    assert first.commentary.startswith("La diagnosi di acalasia")
    assert first.commentary.endswith("Seconda riga del commento.")


def test_parse_examples_tolerates_plain_format_and_suffix_marker() -> None:
    examples = parse_ssm_examples(SAMPLE)
    second = examples[1]
    assert second.correct_option_id == "B"
    assert second.options[1].text == "seconda"
    assert second.commentary == ""


def test_real_examples_file_parses_five_questions() -> None:
    examples = load_ssm_examples(EXAMPLES_PATH)
    assert len(examples) == 5
    assert all(len(e.options) == 5 for e in examples)
    assert all(e.correct_option_id == "A" for e in examples)
    assert all(e.commentary for e in examples)


def test_missing_file_warns_and_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="medquiz_generator.application.examples"):
        examples = load_ssm_examples(tmp_path / "missing.txt")
    assert examples == []
    assert any("few-shot disabled" in record.message for record in caplog.records)


def test_empty_file_warns_and_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("nessuna domanda qui", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="medquiz_generator.application.examples"):
        assert load_ssm_examples(path) == []
    assert any("zero questions" in record.message for record in caplog.records)


def test_format_examples_marks_correct_option() -> None:
    examples = parse_ssm_examples(SAMPLE)
    text = format_examples(examples, max_examples=1)
    assert "Domanda 1" in text
    assert "A: Acalasia esofagea [corretta]" in text
    assert "Domanda 2" not in text
