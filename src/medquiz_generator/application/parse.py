"""Stage 1 — parse: PDF → StructuredDocument (persisted as document.json)."""

from __future__ import annotations

from pathlib import Path

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import Stage, StructuredDocument
from medquiz_generator.domain.ports import DocumentParser


class ParseStage:
    def __init__(self, parser: DocumentParser, *, extract_figures: bool = False) -> None:
        self._parser = parser
        self._extract_figures = extract_figures

    def run(self, ctx: RunContext, pdf_path: Path, *, use_cache: bool = True) -> StructuredDocument:
        with ctx.stage(Stage.PARSE):
            assets_dir = (
                ctx.store.run_dir(ctx.run_id) / artifacts.FIGURES_DIR
                if self._extract_figures
                else None
            )
            document = self._parser.parse(pdf_path, use_cache=use_cache, assets_dir=assets_dir)
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
                figures=len(document.figures),
                content_hash=document.content_hash[:12],
            )
        return document
