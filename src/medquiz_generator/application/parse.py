"""Stage 1 — parse: PDF → StructuredDocument (persisted as document.json)."""

from __future__ import annotations

from pathlib import Path

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import Stage, StructuredDocument
from medquiz_generator.domain.ports import DocumentParser


class ParseStage:
    def __init__(self, parser: DocumentParser) -> None:
        self._parser = parser

    def run(self, ctx: RunContext, pdf_path: Path, *, use_cache: bool = True) -> StructuredDocument:
        with ctx.stage(Stage.PARSE):
            document = self._parser.parse(pdf_path, use_cache=use_cache)
            ctx.store.save(ctx.run_id, artifacts.DOCUMENT, document)
            ctx.update_meta(
                source_path=str(pdf_path),
                document_hash=document.content_hash,
                parser=self._parser.name,
            )
            ctx.log(
                "document parsed",
                sections=len(document.sections),
                pages=document.page_count,
                chars=document.total_chars,
                content_hash=document.content_hash[:12],
            )
        return document
