"""Stage 3 — classify: build the run taxonomy (subjects → topics → sections) from the text.

There is no predefined taxonomy. The builder port extracts it from the document (structure +
LLM normalisation); this stage guarantees completeness — every section with text ends up
mapped — by filling gaps with the structural fallback (chapter → section title).
"""

from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    Section,
    Stage,
    StructuredDocument,
    Taxonomy,
    TaxonomySubject,
    TaxonomyTopic,
)
from medquiz_generator.domain.ports import TaxonomyBuilder


def chapter_of(section: Section, document: StructuredDocument) -> str:
    """Title of the level-1 ancestor (or the section itself when it is a top-level node)."""
    node: Section | None = section
    while node is not None and node.level > 1:
        node = document.section_by_id(node.parent_id) if node.parent_id else None
    return node.title if node is not None else section.title


def structural_taxonomy(document: StructuredDocument, run_id: str) -> Taxonomy:
    """Deterministic scaffold: subject = chapter, topic = section title. Used as fallback
    and as the input outline shown to the LLM normaliser."""
    subjects: dict[str, TaxonomySubject] = {}
    for section in document.sections:
        if not section.text.strip():
            continue
        chapter = chapter_of(section, document)
        subject = subjects.setdefault(chapter, TaxonomySubject(name=chapter))
        subject.topics.append(TaxonomyTopic(name=section.title, section_ids=[section.id]))
    return Taxonomy(
        run_id=run_id, document_hash=document.content_hash, subjects=list(subjects.values())
    )


def ensure_complete(taxonomy: Taxonomy, document: StructuredDocument) -> tuple[Taxonomy, int]:
    """Make the taxonomy a partition of the document's sections.

    * ids unknown to the document are dropped;
    * a section listed under several topics keeps only its first occurrence (``label_for``
      already resolves to the first, so the file now says what the code does);
    * sections the builder left out are filled with the structural fallback (topic = section
      title) under the taxonomy's only subject when there is exactly one, otherwise under the
      chapter title.
    Returns the completed taxonomy and the number of sections that were filled in.
    """
    valid_ids = {s.id for s in document.sections}
    seen: set[str] = set()
    subjects: list[TaxonomySubject] = []
    for subject in taxonomy.subjects:
        topics: list[TaxonomyTopic] = []
        for topic in subject.topics:
            kept: list[str] = []
            for sid in topic.section_ids:
                if sid in valid_ids and sid not in seen:
                    seen.add(sid)
                    kept.append(sid)
            topics.append(TaxonomyTopic(name=topic.name, section_ids=kept))
        subjects.append(TaxonomySubject(name=subject.name, topics=topics))
    completed = taxonomy.model_copy(update={"subjects": subjects})
    filled = 0
    for section in document.sections:
        if not section.text.strip() or section.id in seen:
            continue
        if len(completed.subjects) == 1:
            target = completed.subjects[0]
        else:
            chapter = chapter_of(section, document)
            found = next((s for s in completed.subjects if s.name == chapter), None)
            if found is None:
                found = TaxonomySubject(name=chapter)
                completed.subjects.append(found)
            target = found
        target.topics.append(TaxonomyTopic(name=section.title, section_ids=[section.id]))
        seen.add(section.id)
        filled += 1
    completed.subjects = [
        s.model_copy(update={"topics": [t for t in s.topics if t.section_ids]})
        for s in completed.subjects
    ]
    completed.subjects = [s for s in completed.subjects if s.topics]
    return completed, filled


class ClassifyStage:
    def __init__(self, builder: TaxonomyBuilder) -> None:
        self._builder = builder

    def run(self, ctx: RunContext) -> Taxonomy:
        with ctx.stage(Stage.CLASSIFY):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            taxonomy = self._builder.build(document, ctx.run_id)
            taxonomy, filled = ensure_complete(taxonomy, document)
            ctx.store.save(ctx.run_id, artifacts.TAXONOMY, taxonomy)
            ctx.log(
                "taxonomy built",
                subjects=len(taxonomy.subjects),
                topics=sum(len(s.topics) for s in taxonomy.subjects),
                sections=taxonomy.section_count,
                filled_by_fallback=filled,
                model=self._builder.model_name,
            )
        return taxonomy
