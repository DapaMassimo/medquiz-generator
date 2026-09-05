"""Pipeline contract: parse → plan → classify → generate → verify → dedup → export → report,
entirely on fakes (no network), through the real Container wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medquiz_generator.application import artifacts
from medquiz_generator.domain.models import (
    QuestionStatus,
    QuizItem,
    RunReport,
    StageTiming,
    UsageRecord,
)
from tests.fakes.fake_container import FakeContainer, make_settings

pytestmark = pytest.mark.integration


def test_full_pipeline_on_fakes(tmp_path: Path, fixture_document) -> None:  # type: ignore[no-untyped-def]
    container = FakeContainer(make_settings(tmp_path, plan_min_unit_chars=400), fixture_document)
    pdf = tmp_path / "manuale.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    run_id = container.new_run_id()
    ctx = container.context(run_id)

    report = container.pipeline(ctx).run(ctx, pdf, 15)

    assert isinstance(report, RunReport)
    assert report.total_questions_requested == 15
    assert report.generated == 15
    assert report.discarded == 0
    # FakeGenerator repeats the same quote for questions in the same unit → dedup flags them.
    assert report.verified + report.flagged == 15
    assert report.units_planned == report.units_covered
    assert report.exported == report.verified
    assert report.usage_total.calls > 0 and report.usage_total.cost_usd > 0
    assert report.cost_per_delivered_question_usd is not None
    assert {"generate", "verify", "dedup"} <= set(report.usage_by_stage)
    assert [t.stage for t in report.stage_timings] == [
        "parse",
        "plan",
        "classify",
        "generate",
        "verify",
        "dedup",
        "export",
    ]

    run_dir = container.store.run_dir(run_id)
    for name in (
        artifacts.DOCUMENT,
        artifacts.PLAN,
        artifacts.TAXONOMY,
        artifacts.QUESTIONS_GENERATED,
        artifacts.QUESTIONS_VERIFIED,
        artifacts.QUESTIONS_DEDUPED,
        artifacts.VERIFICATION_REPORT,
        artifacts.DEDUP_REPORT,
        artifacts.RUN_REPORT,
        artifacts.RUN_META,
    ):
        assert (run_dir / f"{name}.json").exists(), name
    assert (run_dir / "usage.jsonl").exists()
    assert (run_dir / "timings.jsonl").exists()
    exported = json.loads((run_dir / "export" / "quiz.json").read_text(encoding="utf-8"))
    assert all(item["status"]["status"] == "verified" for item in exported)
    assert (run_dir / "export" / "quiz.xlsx").exists()

    # Every artefact carries the run id (traceability input → question).
    items = container.store.load_list(run_id, artifacts.QUESTIONS_DEDUPED, QuizItem)
    assert all(i.model_metadata.run_id == run_id for i in items)
    assert all(r.run_id == run_id for r in container.store.read_jsonl(run_id, "usage", UsageRecord))
    assert all(
        isinstance(t, StageTiming)
        for t in container.store.read_jsonl(run_id, "timings", StageTiming)
    )
    assert container.store.latest_run_id() == run_id


def test_stages_are_rerunnable_from_persisted_artifacts(tmp_path: Path, fixture_document) -> None:  # type: ignore[no-untyped-def]
    container = FakeContainer(
        make_settings(tmp_path, verify_judge_enabled=False, dedup_enabled=False), fixture_document
    )
    pdf = tmp_path / "manuale.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    run_id = container.new_run_id()
    ctx = container.context(run_id)
    container.parse_stage().run(ctx, pdf)
    container.plan_stage().run(ctx, 6)
    container.classify_stage(ctx).run(ctx)
    # Re-create the context (as a new CLI invocation would) and continue from disk.
    ctx2 = container.context(run_id)
    items = container.generate_stage(ctx2).run(ctx2)
    verified = container.verify_stage(ctx2).run(ctx2)
    assert len(items) == len(verified) == 6
    assert all(i.status.status == QuestionStatus.VERIFIED for i in verified)
    assert container.judge(ctx2.tracker) is None
    assert container.embeddings(ctx2.tracker) is None
    deduped = container.dedup_stage(ctx2).run(ctx2)
    assert len(deduped) == 6
    written = container.export_stage().run(ctx2)
    assert written["xlsx"].exists()
    report = container.report_stage().run(ctx2)
    assert report.verified == 6 and report.usage_by_stage.keys() == {"generate"}
