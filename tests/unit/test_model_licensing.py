"""The model allowlist is a licence control, so it gets tests like any other rule."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from medquiz_generator.adapters.parsing.docling_parser import (
    ALLOWED_OCR_ENGINES,
    _assert_allowlisted,
    _ocr_options,
)
from medquiz_generator.domain.errors import ParsingError


def _options(table_kind: str, ocr_kind: str = "rapidocr") -> SimpleNamespace:
    return SimpleNamespace(
        table_structure_options=SimpleNamespace(kind=table_kind),
        ocr_options=SimpleNamespace(kind=ocr_kind),
    )


def test_tableformer_v1_passes() -> None:
    _assert_allowlisted(_options("docling_tableformer"), do_ocr=False)


def test_tableformer_v2_is_rejected() -> None:
    """V2 ships no licence on its model card; a default flip must not slip through."""
    with pytest.raises(ParsingError, match="docling_tableformer_v2"):
        _assert_allowlisted(_options("docling_tableformer_v2"), do_ocr=False)


def test_unreviewed_ocr_engine_is_rejected() -> None:
    with pytest.raises(ParsingError, match="not allowlisted"):
        _assert_allowlisted(_options("docling_tableformer", "chandra"), do_ocr=True)


def test_ocr_engine_is_only_checked_when_ocr_runs() -> None:
    _assert_allowlisted(_options("docling_tableformer", "chandra"), do_ocr=False)


@pytest.mark.parametrize("engine", ["chandra", "nemotron-ocr", "auto", ""])
def test_ocr_options_rejects_unlisted_engine(engine: str) -> None:
    with pytest.raises(ParsingError, match="not allowlisted"):
        _ocr_options(engine)


def test_allowlist_contents_are_pinned() -> None:
    """Widening this set is a licence decision, not a refactor."""
    assert ALLOWED_OCR_ENGINES == ("rapidocr", "easyocr", "tesseract")
