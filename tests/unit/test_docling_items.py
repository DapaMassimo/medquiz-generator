"""iter_outline_items on duck-typed Docling items (no Docling needed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medquiz_generator.adapters.parsing.docling_parser import iter_outline_items


@dataclass
class Label:
    value: str


@dataclass
class Prov:
    page_no: int


@dataclass
class Item:
    label: Label
    text: str = ""
    level: int = 1
    prov: list[Prov] = field(default_factory=lambda: [Prov(1)])

    def export_to_markdown(self, doc: Any) -> str:
        return "| a | b |\n|---|---|\n| 1 | 2 |"


class FakeDoc:
    def __init__(self, items: list[Item]) -> None:
        self._items = items

    def iterate_items(self):  # type: ignore[no-untyped-def]
        for item in self._items:
            yield item, 0


def test_captions_are_text_and_caption_like_headings_are_demoted() -> None:
    doc = FakeDoc(
        [
            Item(Label("section_header"), "Definizione e classificazione"),
            Item(Label("text"), "Corpo."),
            Item(Label("section_header"), "Tabella 3.5 Esami di laboratorio"),
            Item(Label("caption"), "Figura 3.1 Schema del sistema RAA"),
            Item(Label("section_header"), "Figura 3.2", prov=[Prov(2)]),
            Item(Label("table"), "", prov=[Prov(2)]),
            Item(Label("picture"), "ignored"),
            Item(Label("page_footer"), "12"),
            Item(Label("section_header"), "Diagnosi", level=1, prov=[Prov(3)]),
        ]
    )
    items = list(iter_outline_items(doc))
    kinds = [(i.kind, i.text.splitlines()[0][:20], i.page) for i in items]
    assert kinds == [
        ("heading", "Definizione e classi", 1),
        ("text", "Corpo.", 1),
        ("text", "Tabella 3.5 Esami di", 1),
        ("text", "Figura 3.1 Schema de", 1),
        ("text", "Figura 3.2", 2),
        ("text", "| a | b |", 2),
        ("heading", "Diagnosi", 3),
    ]
    assert [i.level for i in items if i.kind == "heading"] == [2, 2]
