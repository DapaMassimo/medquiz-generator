from __future__ import annotations

from medquiz_generator.adapters.parsing.outline import (
    OutlineItem,
    build_sections,
    infer_levels_from_case,
)


def test_build_sections_nests_headings_and_tracks_pages() -> None:
    items = [
        OutlineItem("text", "Prefazione.", 1),
        OutlineItem("heading", "Capitolo 1", 1, level=1),
        OutlineItem("text", "Intro capitolo.", 1),
        OutlineItem("heading", "1.1 Sezione", 2, level=2),
        OutlineItem("text", "Testo   sezione uno.", 2),
        OutlineItem("text", "Continua.", 3),
        OutlineItem("heading", "1.2 Sezione", 3, level=2),
        OutlineItem("text", "Testo sezione due.", 3),
        OutlineItem("heading", "Capitolo 2", 4, level=1),
        OutlineItem("text", "Testo capitolo due.", 4),
    ]
    sections = build_sections(items, root_title="Doc")
    titles = [s.title for s in sections]
    assert titles == ["Doc", "Capitolo 1", "1.1 Sezione", "1.2 Sezione", "Capitolo 2"]
    assert [s.level for s in sections] == [0, 1, 2, 2, 1]
    sub = sections[2]
    assert sub.path == ["Doc", "Capitolo 1", "1.1 Sezione"]
    assert sub.parent_id == sections[1].id
    assert (sub.page_start, sub.page_end) == (2, 3)
    assert sub.text == "Testo sezione uno.\n\nContinua."
    assert sections[4].parent_id == sections[0].id


def test_build_sections_normalises_levels_and_skips_empty_root() -> None:
    items = [
        OutlineItem("heading", "Solo un titolo", 1, level=3),
        OutlineItem("text", "Corpo.", 1),
        OutlineItem("heading", "Sotto", 1, level=4),
        OutlineItem("text", "Corpo sotto.", 2),
    ]
    sections = build_sections(items, root_title="Root")
    assert [s.title for s in sections] == ["Solo un titolo", "Sotto"]
    assert [s.level for s in sections] == [1, 2]
    assert sections[0].parent_id is None
    assert sections[1].parent_id == sections[0].id


def test_build_sections_empty_input() -> None:
    assert build_sections([], root_title="Root") == []


def test_infer_levels_from_case_recovers_two_levels() -> None:
    items = [
        OutlineItem("heading", "DIAGNOSI", 1, level=1),
        OutlineItem("heading", "Indicatori diagnostici", 1, level=1),
        OutlineItem("text", "Testo.", 1),
        OutlineItem("heading", "TERAPIA", 2, level=1),
    ]
    inferred = infer_levels_from_case(items)
    assert [i.level for i in inferred if i.kind == "heading"] == [1, 2, 1]
    sections = build_sections(inferred, root_title="Doc")
    assert sections[1].parent_id == sections[0].id and sections[1].level == 2


def test_infer_levels_from_case_leaves_structured_or_uniform_outlines() -> None:
    structured = [
        OutlineItem("heading", "A", 1, level=1),
        OutlineItem("heading", "b", 1, level=2),
    ]
    assert infer_levels_from_case(structured) == structured
    uniform = [OutlineItem("heading", "A", 1, level=1), OutlineItem("heading", "B", 2, level=1)]
    assert infer_levels_from_case(uniform) == uniform
    assert infer_levels_from_case([]) == []
