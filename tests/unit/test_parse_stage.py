from __future__ import annotations

from pathlib import Path

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.parse import ParseStage
from medquiz_generator.domain.models import StructuredDocument
from tests.fakes.fake_parser import FakeParser


def test_parse_stage_persists_document_and_meta(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    parser = FakeParser(fixture_document)
    document = ParseStage(parser).run(ctx, Path("manuale.pdf"), use_cache=False)
    assert parser.calls == [(Path("manuale.pdf"), False)]
    stored = ctx.store.load(ctx.run_id, artifacts.DOCUMENT, StructuredDocument)
    assert stored.content_hash == document.content_hash
    assert len(stored.sections) == len(fixture_document.sections)
    meta = ctx.meta()
    assert meta.document_hash == document.content_hash
    assert meta.parser == "fake-parser"
    assert meta.source_path == "manuale.pdf"
