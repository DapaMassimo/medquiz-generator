from __future__ import annotations

import pytest

from medquiz_generator.application import artifacts
from medquiz_generator.application.classify import ClassifyStage
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.generate import GenerateStage, pick_source_section
from medquiz_generator.application.plan import PlanStage
from medquiz_generator.domain.errors import StageInputMissingError
from medquiz_generator.domain.models import (
    GeneratedQuestion,
    GenerationPlan,
    QuizItem,
    StructuredDocument,
    UsageRecord,
)
from tests.fakes.fake_llm import FakeGenerator, FakeTaxonomyBuilder


def prepare(ctx: RunContext, document: StructuredDocument, total: int) -> GenerationPlan:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, document)
    plan = PlanStage(min_unit_chars=400).run(ctx, total)
    ClassifyStage(FakeTaxonomyBuilder()).run(ctx)
    return plan


def test_generate_requires_taxonomy(ctx: RunContext, fixture_document: StructuredDocument) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    PlanStage(min_unit_chars=400).run(ctx, 5)
    with pytest.raises(StageInputMissingError):
        GenerateStage(FakeGenerator(), []).run(ctx)


def test_generate_produces_planned_count_with_labels_and_usage(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    plan = prepare(ctx, fixture_document, 12)
    generator = FakeGenerator(ctx.tracker)
    items = GenerateStage(generator, []).run(ctx)
    assert len(items) == plan.planned_questions == 12
    assert [call[0] for call in generator.calls] == [u.id for u in plan.units if u.questions]
    stored = ctx.store.load_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, QuizItem)
    assert [i.id for i in stored] == [i.id for i in items]
    for item in items:
        assert item.materia and item.argomento
        assert item.source.section_id in {s.id for s in fixture_document.sections}
        assert item.model_metadata.run_id == ctx.run_id
        assert item.status.status == "generated"
    usage = ctx.store.read_jsonl(ctx.run_id, artifacts.USAGE_LOG, UsageRecord)
    assert len(usage) == len(generator.calls)
    assert all(r.stage == "generate" and r.unit_id for r in usage)
    assert all(r.cost_usd > 0 for r in usage)


def test_generate_is_resumable_per_unit(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    plan = prepare(ctx, fixture_document, 8)
    first_unit = plan.units[0].id
    generator = FakeGenerator()
    partial = GenerateStage(generator, []).run(ctx, unit_ids=[first_unit])
    assert {i.model_metadata.unit_id for i in partial} == {first_unit}
    resumed = GenerateStage(generator, []).run(ctx)
    assert len(resumed) == plan.planned_questions
    # The first unit was not generated twice.
    assert [c[0] for c in generator.calls].count(first_unit) == 1


def test_pick_source_section_prefers_quote_then_page(
    fixture_document: StructuredDocument,
) -> None:
    sections = [fixture_document.sections[7], fixture_document.sections[8]]  # s007, s008
    quoted = GeneratedQuestion(
        question="q",
        options=[],
        correct_option_id="A",
        explanation="e",
        difficulty="media",
        quote="la metformina è il farmaco di prima scelta",
        page=5,
    )
    assert pick_source_section(quoted, sections).id == "s008"
    by_page = quoted.model_copy(update={"quote": "testo assente", "page": 6})
    assert pick_source_section(by_page, sections).id == "s008"
    fallback = quoted.model_copy(update={"quote": "testo assente", "page": None})
    assert pick_source_section(fallback, sections).id == "s007"
