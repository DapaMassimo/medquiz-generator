from __future__ import annotations

import json
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


def test_real_examples_file_parses_all_questions() -> None:
    examples = load_ssm_examples(EXAMPLES_PATH)
    assert [e.number for e in examples] == list(range(1, 20))
    assert all(len(e.options) == 5 for e in examples)
    assert all([o.id for o in e.options] == ["A", "B", "C", "D", "E"] for e in examples)
    # Batches are interleaved so that the first examples do not all share the same answer.
    key = "".join(e.correct_option_id or "?" for e in examples)
    assert key == "CADBADBAEDABCAAAEDC"
    assert len(set(key[:5])) >= 3
    assert all(e.commentary for e in examples)
    assert all(len(e.question) > 10 for e in examples)  # shortest stem: "La dopamina:"


MARKDOWN_SAMPLE = """# Quiz SSM — Domande con correzione commentata

---

## Domanda 1

Un giovane di 28 anni si presenta con pomfi pruriginosi dopo esercizio fisico.
Come si definisce tale manifestazione?

- A. Angioedema ereditario
- B. Dermatografismo
- **C. Orticaria colinergica** ✅
- D. Malattia da siero
- E. Vasculite necrotizzante

**Spiegazione:** L'orticaria colinergica è un tipo di orticaria fisica.
Seconda riga della spiegazione.

---

## Domanda 2

**Scenario 1.2** — Qual è l'antidoto da somministrare?

- **A. Intralipid 20%** ✅
- B. Soluzione glucosata 10%
- C. Soluzione fisiologica
- D. Soluzione glucosata 33%
- E. Soluzione di Ringer lattato

**Spiegazione:** Emulsioni lipidiche.

---

## Riepilogo risposte

| # | Risposta |
|---|----------|
| 1 | C |
| 2 | A |
"""


def test_parse_markdown_dump_format() -> None:
    examples = parse_ssm_examples(MARKDOWN_SAMPLE)
    assert [e.number for e in examples] == [1, 2]
    first, second = examples
    assert first.question.startswith("Un giovane di 28 anni") and first.question.endswith("?")
    assert [o.id for o in first.options] == ["A", "B", "C", "D", "E"]
    assert first.correct_option_id == "C"
    assert first.options[2].text == "Orticaria colinergica"  # no ** or ✅ left behind
    assert first.commentary == (
        "L'orticaria colinergica è un tipo di orticaria fisica. Seconda riga della spiegazione."
    )
    assert second.question == "Scenario 1.2 — Qual è l'antidoto da somministrare?"
    assert second.correct_option_id == "A" and second.options[0].text == "Intralipid 20%"
    # The trailing answer-key table is not swallowed into the last commentary.
    assert second.commentary == "Emulsioni lipidiche."
    assert "Riepilogo" not in second.commentary


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


def test_format_examples_keeps_original_option_order() -> None:
    examples = load_ssm_examples(EXAMPLES_PATH)
    text = format_examples(examples, max_examples=3)
    marked = [line[0] for line in text.splitlines() if "[corretta]" in line]
    assert marked == [e.correct_option_id for e in examples[:3]]
