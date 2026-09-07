from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.classify import (
    ClassifyStage,
    chapter_of,
    ensure_complete,
    structural_taxonomy,
)
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    StructuredDocument,
    Taxonomy,
    TaxonomySubject,
    TaxonomyTopic,
)
from tests.fakes.fake_llm import FakeTaxonomyBuilder


def test_structural_taxonomy_maps_every_text_section(fixture_document: StructuredDocument) -> None:
    taxonomy = structural_taxonomy(fixture_document, "r")
    with_text = [s for s in fixture_document.sections if s.text.strip()]
    assert taxonomy.section_count == len(with_text)
    label = taxonomy.label_for("s003")
    assert label is not None
    assert label.subject == "Capitolo 1 - Ipertensione arteriosa"
    assert label.topic == "1.2 Diagnosi"
    assert chapter_of(fixture_document.sections[0], fixture_document) == fixture_document.title


def test_ensure_complete_fills_missing_and_drops_unknown_ids(
    fixture_document: StructuredDocument,
) -> None:
    partial = Taxonomy(
        run_id="r",
        document_hash="h",
        subjects=[
            TaxonomySubject(
                name="Cardiologia",
                topics=[TaxonomyTopic(name="Ipertensione", section_ids=["s002", "s003", "ghost"])],
            )
        ],
    )
    completed, filled = ensure_complete(partial, fixture_document)
    assert filled == len([s for s in fixture_document.sections if s.text.strip()]) - 2
    assert completed.label_for("ghost") is None
    # With a single subject, filled sections join it instead of creating a chapter subject.
    label = completed.label_for("s004")
    assert label is not None and (label.subject, label.topic) == ("Cardiologia", "1.3 Terapia")
    assert [s.name for s in completed.subjects] == ["Cardiologia"]
    for section in fixture_document.sections:
        if section.text.strip():
            assert completed.label_for(section.id) is not None


def test_ensure_complete_uses_chapter_when_several_subjects(
    fixture_document: StructuredDocument,
) -> None:
    partial = Taxonomy(
        run_id="r",
        document_hash="h",
        subjects=[
            TaxonomySubject(name="A", topics=[TaxonomyTopic(name="a", section_ids=["s002"])]),
            TaxonomySubject(name="B", topics=[TaxonomyTopic(name="b", section_ids=["s006"])]),
        ],
    )
    completed, _ = ensure_complete(partial, fixture_document)
    label = completed.label_for("s004")
    assert label is not None and label.subject == "Capitolo 1 - Ipertensione arteriosa"


def test_ensure_complete_keeps_first_occurrence_of_duplicated_ids(
    fixture_document: StructuredDocument,
) -> None:
    duplicated = Taxonomy(
        run_id="r",
        document_hash="h",
        subjects=[
            TaxonomySubject(
                name="Cardiologia",
                topics=[
                    TaxonomyTopic(name="Diagnosi", section_ids=["s003", "s004"]),
                    TaxonomyTopic(name="Terapia", section_ids=["s004", "s008"]),
                ],
            )
        ],
    )
    completed, _ = ensure_complete(duplicated, fixture_document)
    ids = [sid for s in completed.subjects for t in s.topics for sid in t.section_ids]
    assert len(ids) == len(set(ids))
    label = completed.label_for("s004")
    assert label is not None and label.topic == "Diagnosi"
    assert completed.section_count == len([s for s in fixture_document.sections if s.text.strip()])


def test_classify_stage_persists_taxonomy(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    taxonomy = ClassifyStage(FakeTaxonomyBuilder()).run(ctx)
    stored = ctx.store.load(ctx.run_id, artifacts.TAXONOMY, Taxonomy)
    assert stored.section_count == taxonomy.section_count
    assert stored.source == "extracted"
