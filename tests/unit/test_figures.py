from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from medquiz_generator.adapters.parsing.docling_parser import (
    iter_outline_items,
    structured_document_from_docling,
)
from medquiz_generator.adapters.parsing.outline import OutlineItem, build_outline
from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.figures import FiguresStage
from medquiz_generator.application.parse import ParseStage
from medquiz_generator.domain.models import FigureCatalog, FigureRef, StructuredDocument
from tests.fakes.fake_llm import FakeFigureTriage
from tests.fakes.fake_parser import FakeParser

# 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_build_outline_attaches_figures_to_the_current_section() -> None:
    items = [
        OutlineItem("heading", "Capitolo 1", 1, level=1),
        OutlineItem("text", "Intro.", 1),
        OutlineItem(
            "figure", "Figura 1.1 ECG", 1, ref="assets/figures/f001.png", width=300, height=200
        ),
        OutlineItem("heading", "1.1 Sezione", 2, level=2),
        OutlineItem("figure", "Figura 1.2 Schema", 2, ref="assets/figures/f002.png"),
        OutlineItem("text", "Testo.", 2),
    ]
    sections, figures = build_outline(items, root_title="Doc")
    assert [f.id for f in figures] == ["f001", "f002"]
    assert figures[0].section_id == sections[0].id and figures[0].caption == "Figura 1.1 ECG"
    assert figures[0].width == 300 and figures[0].path == "assets/figures/f001.png"
    assert figures[1].section_id == sections[1].id


@dataclass
class Label:
    value: str


@dataclass
class BBox:
    l: float  # noqa: E741 - Docling's own field names
    t: float
    r: float
    b: float


@dataclass
class Prov:
    page_no: int
    bbox: BBox


class FakeImage:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(PNG)


@dataclass
class Item:
    label: Label
    text: str = ""
    level: int = 1
    prov: list[Prov] = field(default_factory=lambda: [Prov(1, BBox(0, 200, 300, 0))])
    caption: str = ""

    def get_image(self, doc: Any) -> FakeImage | None:
        return FakeImage()

    def caption_text(self, doc: Any) -> str:
        return self.caption


class FakeDoc:
    pages: ClassVar[dict[int, None]] = {1: None, 2: None}

    def __init__(self, items: list[Item]) -> None:
        self._items = items

    def iterate_items(self):  # type: ignore[no-untyped-def]
        for item in self._items:
            yield item, 0


def test_docling_pictures_become_assets_and_icons_are_skipped(tmp_path: Path) -> None:
    doc = FakeDoc(
        [
            Item(Label("section_header"), "Capitolo"),
            Item(Label("text"), "Corpo del capitolo."),
            Item(Label("picture"), caption="Figura 3.1 Radiografia del torace"),
            Item(Label("picture"), prov=[Prov(1, BBox(0, 40, 40, 0))], caption="icona"),
        ]
    )
    assets = tmp_path / "run" / "assets" / "figures"
    document = structured_document_from_docling(
        doc, source_path=Path("x.pdf"), content_hash="h", parser_name="fake", assets_dir=assets
    )
    assert len(document.figures) == 1
    figure = document.figures[0]
    assert figure.path == "assets/figures/f001.png" and (assets / "f001.png").read_bytes() == PNG
    assert figure.caption.startswith("Figura 3.1") and figure.section_id == document.sections[0].id
    # Without an assets dir, pictures are ignored as before.
    plain = list(iter_outline_items(doc))
    assert all(i.kind != "figure" for i in plain)


def test_parse_stage_passes_assets_dir_only_when_enabled(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    parser = FakeParser(fixture_document)
    ParseStage(parser).run(ctx, Path("a.pdf"))
    ParseStage(parser, extract_figures=True).run(ctx, Path("a.pdf"))
    assert parser.assets_dirs[0] is None
    assert parser.assets_dirs[1] == ctx.store.run_dir(ctx.run_id) / artifacts.FIGURES_DIR


def test_figures_stage_triages_and_is_resumable(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    run_dir = ctx.store.run_dir(ctx.run_id)
    (run_dir / "assets" / "figures").mkdir(parents=True)
    for name in ("f001.png", "f002.png"):
        (run_dir / "assets" / "figures" / name).write_bytes(PNG)
    figures = [
        FigureRef(
            id="f001",
            page=2,
            section_id="s003",
            caption="Figura 1 ECG a 12 derivazioni",
            path="assets/figures/f001.png",
        ),
        FigureRef(
            id="f002",
            page=3,
            section_id="s004",
            caption="Figura 2 Schema terapeutico",
            path="assets/figures/f002.png",
        ),
        FigureRef(
            id="f003",
            page=3,
            section_id="s004",
            caption="missing file",
            path="assets/figures/f003.png",
        ),
    ]
    ctx.store.save(
        ctx.run_id, artifacts.DOCUMENT, fixture_document.model_copy(update={"figures": figures})
    )
    triage = FakeFigureTriage(ctx.tracker)
    catalog = FiguresStage(triage).run(ctx)
    by_id = {f.id: f for f in catalog.figures}
    assert by_id["f001"].kind == "clinical_image" and by_id["f001"].question_worthy
    assert by_id["f002"].kind == "diagram" and not by_id["f002"].question_worthy
    assert by_id["f003"].kind == "unclassified"  # no image file → left untouched
    assert triage.calls == ["f001", "f002"]
    assert ctx.store.load(ctx.run_id, artifacts.FIGURES, FigureCatalog).figures == catalog.figures
    # Re-running reuses the triage of classified figures.
    FiguresStage(triage).run(ctx)
    assert triage.calls == ["f001", "f002", "f003"] or triage.calls == ["f001", "f002"]
    usage = [r for r in ctx.tracker.records() if r.stage == "figures"]
    assert usage and usage[0].item_id == "f001"
