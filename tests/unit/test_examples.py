from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from medquiz_generator.application.examples import (
    format_examples,
    load_ssm_examples,
    parse_ssm_examples,
    rotate_correct_option,
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


def test_rotate_correct_option_moves_answer_and_relabels() -> None:
    [example] = parse_ssm_examples(SAMPLE)[:1]
    rotated = rotate_correct_option(example, 3)
    assert rotated.correct_option_id == "D"
    assert [o.id for o in rotated.options] == ["A", "B", "C", "D", "E"]
    assert rotated.options[3].text == "Acalasia esofagea"
    # Cyclic rotation: the relative order of the other options is preserved.
    assert [o.text for o in rotated.options] == [
        "Malformazione",
        "Sindrome di Mallory-Weiss",
        "Neoplasia",
        "Acalasia esofagea",
        "Stenosi organica",
    ]
    assert example.correct_option_id == "A"  # the original is untouched


def test_format_examples_rotates_correct_position_across_examples() -> None:
    examples = load_ssm_examples(EXAMPLES_PATH)
    assert all(e.correct_option_id == "A" for e in examples)  # the raw file is A-biased
    text = format_examples(examples)
    positions = [line[0] for line in text.splitlines() if "[corretta]" in line]
    assert positions == ["A", "B", "C", "D", "E"]


def test_load_structured_json_examples(tmp_path: Path) -> None:
    examples = parse_ssm_examples(SAMPLE)
    path = tmp_path / "examples.json"
    path.write_text(json.dumps([e.model_dump(mode="json") for e in examples]), encoding="utf-8")
    loaded = load_ssm_examples(path)
    assert loaded == examples


def test_real_json_examples_match_text_file() -> None:
    from_text = load_ssm_examples(EXAMPLES_PATH)
    from_json = load_ssm_examples(EXAMPLES_PATH.with_suffix(".json"))
    assert from_json == from_text
