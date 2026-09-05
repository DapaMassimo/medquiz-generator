"""Pure outline builder: a flat stream of headings/paragraphs → list[Section].

Kept free of Docling types so it can be unit-tested with plain tuples. The Docling adapter
only has to translate its items into ``OutlineItem``s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medquiz_generator.domain.models import Section

_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class OutlineItem:
    kind: str  # "heading" | "text"
    text: str
    page: int
    level: int = 0  # headings only


@dataclass
class _Node:
    title: str
    level: int
    page_start: int
    page_end: int
    parent: _Node | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def path(self) -> list[str]:
        titles: list[str] = []
        node: _Node | None = self
        while node is not None:
            if node.title:
                titles.append(node.title)
            node = node.parent
        return list(reversed(titles))


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def infer_levels_from_case(items: list[OutlineItem]) -> list[OutlineItem]:
    """Recover a two-level outline when the parser flattened every heading to one level.

    Textbooks typically set main sections in ALL CAPS and subsections in mixed case; when all
    headings share one level and both styles occur, caps headings keep the level and mixed-case
    ones become their children. Otherwise items are returned unchanged.
    """
    headings = [i for i in items if i.kind == "heading"]
    if len(headings) < 2 or len({h.level for h in headings}) != 1:
        return items
    caps = [_is_all_caps(h.text) for h in headings]
    if all(caps) or not any(caps):
        return items
    base = headings[0].level
    return [
        OutlineItem("heading", i.text, i.page, level=base if _is_all_caps(i.text) else base + 1)
        if i.kind == "heading"
        else i
        for i in items
    ]


def build_sections(items: list[OutlineItem], *, root_title: str) -> list[Section]:
    """Build the section list in document order.

    * Text before the first heading goes into a level-0 root section titled ``root_title``
      (only emitted if it has text).
    * Heading levels are normalised so the shallowest heading is level 1.
    * A section's ``page_end`` is the last page of its own text (or its heading page).
    """
    headings = [i for i in items if i.kind == "heading"]
    min_level = min((h.level for h in headings), default=1)
    shift = 1 - min_level

    root = _Node(title=root_title, level=0, page_start=1, page_end=1, parent=None)
    nodes: list[_Node] = [root]
    stack: list[_Node] = [root]
    first_page_seen = False

    for item in items:
        page = max(1, item.page)
        if not first_page_seen:
            root.page_start = page
            root.page_end = page
            first_page_seen = True
        if item.kind == "heading":
            level = item.level + shift
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = _Node(
                title=_clean(item.text) or "Untitled",
                level=level,
                page_start=page,
                page_end=page,
                parent=stack[-1],
            )
            nodes.append(node)
            stack.append(node)
            continue
        text = _clean(item.text)
        if not text:
            continue
        current = stack[-1]
        current.paragraphs.append(text)
        current.page_end = max(current.page_end, page)

    sections: list[Section] = []
    id_by_node: dict[int, str] = {}
    for index, node in enumerate(nodes):
        if node is root and not node.paragraphs:
            continue
        section_id = f"s{index:03d}"
        id_by_node[id(node)] = section_id
        parent_id = id_by_node.get(id(node.parent)) if node.parent is not None else None
        sections.append(
            Section(
                id=section_id,
                title=node.title,
                level=node.level,
                path=node.path,
                parent_id=parent_id,
                page_start=node.page_start,
                page_end=max(node.page_end, node.page_start),
                text="\n\n".join(node.paragraphs),
            )
        )
    return sections
