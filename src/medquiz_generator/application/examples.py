"""Robust parser for the bundled ``ssm_examples.txt`` (free-form real exam questions).

The file is the few-shot style reference for the generator and the *style golden set* for the
eval harness. The format is loose ("Domanda N", option lines with ☑/☐ markers, a "Commento:"
paragraph), so parsing is tolerant: anything that cannot be parsed is skipped, never fatal.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import cast

from medquiz_generator.domain.models import OPTION_IDS, Option, OptionId, SsmExample

_QUESTION_HEADER = re.compile(r"^\s*#*\s*Domanda\s+(\d+)\s*[:.)]?\s*$", re.IGNORECASE)
_OPTION_LINE = re.compile(
    r"^\s*(?P<mark>[☑☐✓✔■□•\-*]?)\s*(?P<id>[A-Ea-e])\s*[:.)\-]\s*(?P<text>.+?)\s*$"
)
_CORRECT_SUFFIX = re.compile(r"\s*(\((?:risposta\s+)?corretta\)|✅|✔|✓)\s*$", re.IGNORECASE)
_COMMENT_PREFIX = re.compile(
    r"^\s*(Commento|Spiegazione|Risposta)\s*:\s*(?P<text>.*)$", re.IGNORECASE
)
# A trailing answer-key / summary section ends the last question's block.
_STOP_LINE = re.compile(r"^\s*#*\s*(Riepilogo|Soluzioni|Risposte)\b", re.IGNORECASE)
_SEPARATOR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_EMPHASIS = re.compile(r"\*\*|__")
_CORRECT_MARKS = {"☑", "✓", "✔", "■"}


def _clean_line(raw: str) -> str:
    """Drop Markdown emphasis markers so '**C. text** ✅' parses like 'C. text ✅'."""
    return _EMPHASIS.sub("", raw).rstrip()


log = logging.getLogger(__name__)


def _parse_block(number: int, lines: list[str]) -> SsmExample | None:
    question_lines: list[str] = []
    options: list[Option] = []
    correct: OptionId | None = None
    commentary_lines: list[str] = []
    mode = "question"
    for raw in lines:
        line = _clean_line(raw)
        if _SEPARATOR.match(line):
            continue
        comment = _COMMENT_PREFIX.match(line)
        if comment:
            mode = "comment"
            if comment.group("text"):
                commentary_lines.append(comment.group("text"))
            continue
        if mode == "comment":
            if _STOP_LINE.match(line):
                break
            if line.strip():
                commentary_lines.append(line.strip())
            continue
        option = _OPTION_LINE.match(line)
        if option and option.group("id").upper() in OPTION_IDS:
            mode = "options"
            option_id = cast(OptionId, option.group("id").upper())
            text = option.group("text")
            is_correct = option.group("mark") in _CORRECT_MARKS or bool(
                _CORRECT_SUFFIX.search(text)
            )
            text = _CORRECT_SUFFIX.sub("", text).strip()
            text = _EMPHASIS.sub("", text).strip()
            options.append(Option(id=option_id, text=text))
            if is_correct and correct is None:
                correct = option_id
            continue
        if mode == "question" and line.strip():
            question_lines.append(line.strip())
    if not question_lines or not options:
        return None
    return SsmExample(
        number=number,
        question=" ".join(question_lines),
        options=options,
        correct_option_id=correct,
        commentary=" ".join(commentary_lines),
    )


def parse_ssm_examples(text: str) -> list[SsmExample]:
    examples: list[SsmExample] = []
    current_number: int | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        header = _QUESTION_HEADER.match(_clean_line(line))
        if header:
            if current_number is not None:
                parsed = _parse_block(current_number, buffer)
                if parsed:
                    examples.append(parsed)
            current_number = int(header.group(1))
            buffer = []
            continue
        if current_number is not None:
            buffer.append(line)
    if current_number is not None:
        parsed = _parse_block(current_number, buffer)
        if parsed:
            examples.append(parsed)
    return examples


def load_ssm_examples(path: Path) -> list[SsmExample]:
    """Load the examples file (free text ``.txt`` or structured ``.json``).

    A missing/empty file yields [] with a warning (never fatal).
    """
    if not path.exists():
        log.warning(
            "SSM examples file not found at %s: generating with the style described in the "
            "system prompt only (few-shot disabled).",
            path,
        )
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = [SsmExample.model_validate(entry) for entry in payload]
    else:
        examples = parse_ssm_examples(path.read_text(encoding="utf-8"))
    if not examples:
        log.warning("SSM examples file %s parsed to zero questions: few-shot disabled.", path)
    return examples


def format_examples(examples: list[SsmExample], max_examples: int | None = None) -> str:
    """Render examples as few-shot text (Italian content, as in the real exam)."""
    chunks: list[str] = []
    for example in examples[:max_examples]:
        lines = [f"Domanda {example.number}", example.question, ""]
        for option in example.options:
            marker = "[corretta]" if option.id == example.correct_option_id else ""
            lines.append(f"{option.id}: {option.text} {marker}".rstrip())
        if example.commentary:
            lines += ["", f"Commento: {example.commentary}"]
        chunks.append("\n".join(lines))
    return "\n\n---\n\n".join(chunks)
