"""Fake DocumentParser returning a pre-built StructuredDocument (no Docling, no PDF)."""

from __future__ import annotations

from pathlib import Path

from medquiz_generator.domain.models import StructuredDocument


class FakeParser:
    def __init__(self, document: StructuredDocument) -> None:
        self._document = document
        self.calls: list[tuple[Path, bool]] = []

    @property
    def name(self) -> str:
        return "fake-parser"

    def parse(self, pdf_path: Path, *, use_cache: bool = True) -> StructuredDocument:
        self.calls.append((pdf_path, use_cache))
        return self._document.model_copy(update={"source_path": str(pdf_path)})
