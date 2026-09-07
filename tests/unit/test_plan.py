from __future__ import annotations

import pytest

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.plan import (
    AUTO_MIN_UNIT_CHARS_CAP,
    AUTO_MIN_UNIT_CHARS_FLOOR,
    PlanStage,
    allocate,
    build_plan,
    choose_min_unit_chars,
)
from medquiz_generator.domain.errors import PlanningError, StageInputMissingError
from medquiz_generator.domain.models import GenerationPlan, StageTiming, StructuredDocument


def test_allocate_sums_to_total_and_respects_minimum() -> None:
    result = allocate(10, [0.5, 0.3, 0.2])
    assert sum(result) == 10
    assert all(r >= 1 for r in result)
    assert result == [5, 3, 2]


def test_allocate_largest_remainder() -> None:
    # 7 questions, 3 units, 4 remaining after minimums: raw = [2.0, 1.2, 0.8]
    assert allocate(7, [0.5, 0.3, 0.2]) == [3, 2, 2]


def test_allocate_total_smaller_than_units_picks_heaviest() -> None:
    assert allocate(2, [0.2, 0.5, 0.3]) == [0, 1, 1]


def test_allocate_empty() -> None:
    assert allocate(5, []) == []


def test_build_plan_covers_every_substantial_section(
    fixture_document: StructuredDocument,
) -> None:
    plan = build_plan(fixture_document, "r1", 20, min_unit_chars=400)
    planned_ids = {sid for unit in plan.units for sid in unit.section_ids}
    substantial = {s.id for s in fixture_document.sections if s.char_count >= 120}
    assert planned_ids == substantial
    assert plan.planned_questions == 20
    assert all(unit.questions >= 1 for unit in plan.units)
    # Bibliography (noise) and the empty "Note" section are skipped, never planned.
    assert set(plan.skipped_section_ids) == {"s009", "s010"}


def test_build_plan_budget_is_proportional_to_length(
    fixture_document: StructuredDocument,
) -> None:
    plan = build_plan(fixture_document, "r1", 30, min_unit_chars=400)
    largest = max(plan.units, key=lambda u: u.char_count)
    assert largest.questions == max(u.questions for u in plan.units)
    smallest = min(plan.units, key=lambda u: u.char_count)
    assert smallest.questions < largest.questions
    assert abs(sum(u.weight for u in plan.units) - 1.0) < 1e-9


def test_build_plan_merges_small_siblings(fixture_document: StructuredDocument) -> None:
    plan = build_plan(fixture_document, "r1", 10, min_unit_chars=400)
    merged = [u for u in plan.units if len(u.section_ids) > 1]
    assert merged, "2.2 and 2.3 are small siblings and must be merged"
    unit = merged[0]
    assert unit.section_ids == ["s007", "s008"]
    assert unit.page_start == 5 and unit.page_end == 6
    assert "merged" in unit.note


def test_build_plan_skips_bibliography_even_when_long(
    fixture_document: StructuredDocument,
) -> None:
    long_bibliography = fixture_document.sections[9].model_copy(update={"text": "Voce. " * 300})
    sections = [*fixture_document.sections[:9], long_bibliography, fixture_document.sections[10]]
    document = fixture_document.model_copy(update={"sections": sections})
    plan = build_plan(document, "r1", 10, min_unit_chars=400)
    assert "s009" in plan.skipped_section_ids
    assert all("s009" not in unit.section_ids for unit in plan.units)


def test_choose_min_unit_chars_gives_every_unit_a_question(
    fixture_document: StructuredDocument,
) -> None:
    # Generous budget: the finest threshold (the floor) already fits.
    assert choose_min_unit_chars(fixture_document, 20) == AUTO_MIN_UNIT_CHARS_FLOOR
    # Tight budget: the threshold grows until units <= questions (merging never crosses
    # parents, so the fixture cannot go below 5 units: root, chapter intros, two sibling runs).
    threshold = choose_min_unit_chars(fixture_document, 5)
    assert AUTO_MIN_UNIT_CHARS_FLOOR < threshold <= AUTO_MIN_UNIT_CHARS_CAP
    plan = build_plan(fixture_document, "r1", 5, min_unit_chars=threshold)
    assert len(plan.units) <= 5 and all(u.questions >= 1 for u in plan.units)
    # Impossible budget: the cap is returned and zero-budget units remain.
    assert choose_min_unit_chars(fixture_document, 1) == AUTO_MIN_UNIT_CHARS_CAP


def test_build_plan_auto_threshold_is_recorded(fixture_document: StructuredDocument) -> None:
    auto = build_plan(fixture_document, "r1", 3)
    assert auto.min_unit_chars_auto is True
    assert auto.min_unit_chars == choose_min_unit_chars(fixture_document, 3)
    fixed = build_plan(fixture_document, "r1", 3, min_unit_chars=400)
    assert fixed.min_unit_chars_auto is False and fixed.min_unit_chars == 400


def test_plan_stage_override_beats_configured_threshold(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    plan = PlanStage(min_unit_chars=None).run(ctx, 4, min_unit_chars=400)
    assert plan.min_unit_chars == 400 and plan.min_unit_chars_auto is False
    plan = PlanStage(min_unit_chars=None).run(ctx, 4)
    assert plan.min_unit_chars_auto is True


def test_build_plan_rejects_non_positive_total(fixture_document: StructuredDocument) -> None:
    with pytest.raises(PlanningError):
        build_plan(fixture_document, "r1", 0)


def test_build_plan_rejects_empty_document(fixture_document: StructuredDocument) -> None:
    empty = fixture_document.model_copy(
        update={"sections": [s.model_copy(update={"text": ""}) for s in fixture_document.sections]}
    )
    with pytest.raises(PlanningError):
        build_plan(empty, "r1", 5)


def test_plan_stage_persists_and_requires_document(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    with pytest.raises(StageInputMissingError):
        PlanStage().run(ctx, 5)
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    plan = PlanStage(min_unit_chars=400).run(ctx, 12)
    stored = ctx.store.load(ctx.run_id, artifacts.PLAN, GenerationPlan)
    assert stored.planned_questions == plan.planned_questions == 12
    assert ctx.meta().total_questions == 12
    timings = ctx.store.read_jsonl(ctx.run_id, artifacts.TIMINGS_LOG, StageTiming)
    assert [(t.stage, t.status) for t in timings] == [("plan", "failed"), ("plan", "ok")]
    assert timings[0].error is not None and timings[0].error.startswith("StageInputMissingError")
    assert timings[1].error is None
