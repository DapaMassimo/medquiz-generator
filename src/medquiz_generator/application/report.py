"""Stage 9 — report: aggregate every artefact of a run into a persisted RunReport.

Costs are aggregated per stage, per unit (section group) and per question; the headline
number is the cost per *delivered* question (verified + flagged), which is what a client
needs to extrapolate to a whole library.
"""

from __future__ import annotations

from collections import Counter

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    DedupReport,
    FigureCatalog,
    GenerationPlan,
    QuestionStatus,
    QuizItem,
    RunReport,
    StageTiming,
    StructuredDocument,
    UsageRecord,
    UsageSummary,
    VerificationReport,
)


def latest_items(ctx: RunContext) -> list[QuizItem]:
    for name in (
        artifacts.QUESTIONS_DEDUPED,
        artifacts.QUESTIONS_VERIFIED,
        artifacts.QUESTIONS_GENERATED,
    ):
        if ctx.store.exists(ctx.run_id, name):
            return ctx.store.load_list(ctx.run_id, name, QuizItem)
    return []


def summarize_usage(
    records: list[UsageRecord],
) -> tuple[dict[str, UsageSummary], dict[str, UsageSummary], dict[str, UsageSummary], UsageSummary]:
    by_stage: dict[str, UsageSummary] = {}
    by_unit: dict[str, UsageSummary] = {}
    by_item: dict[str, UsageSummary] = {}
    total = UsageSummary()
    for record in records:
        by_stage.setdefault(record.stage or "unknown", UsageSummary()).add(record)
        if record.unit_id:
            by_unit.setdefault(record.unit_id, UsageSummary()).add(record)
        if record.item_id:
            by_item.setdefault(record.item_id, UsageSummary()).add(record)
        total.add(record)
    return by_stage, by_unit, by_item, total


def build_run_report(ctx: RunContext) -> RunReport:
    meta = ctx.meta()
    items = latest_items(ctx)
    by_status = Counter(item.status.status for item in items)

    units_planned = 0
    sections_total = 0
    styles_requested: Counter[str] = Counter()
    if ctx.store.exists(ctx.run_id, artifacts.PLAN):
        plan = ctx.store.load(ctx.run_id, artifacts.PLAN, GenerationPlan)
        for unit in plan.units:
            styles_requested.update(unit.styles)
        units_planned = sum(1 for u in plan.units if u.questions > 0)
        sections_total = sum(len(u.section_ids) for u in plan.units if u.questions > 0)
    elif ctx.store.exists(ctx.run_id, artifacts.DOCUMENT):
        document = ctx.store.load(ctx.run_id, artifacts.DOCUMENT, StructuredDocument)
        sections_total = sum(1 for s in document.sections if s.text.strip())

    delivered = [
        i for i in items if i.status.status in (QuestionStatus.VERIFIED, QuestionStatus.FLAGGED)
    ]
    deterministic: Counter[str] = Counter()
    judge_verdicts: Counter[str] = Counter()
    if ctx.store.exists(ctx.run_id, artifacts.VERIFICATION_REPORT):
        verification = ctx.store.load(ctx.run_id, artifacts.VERIFICATION_REPORT, VerificationReport)
        for record in verification.records:
            deterministic.update(record.deterministic_failures)
            if record.judge is not None:
                judge_verdicts[record.judge.verdict] += 1
    duplicates = 0
    if ctx.store.exists(ctx.run_id, artifacts.DEDUP_REPORT):
        duplicates = len(ctx.store.load(ctx.run_id, artifacts.DEDUP_REPORT, DedupReport).pairs)

    figures_total = 0
    figures_worthy = 0
    if ctx.store.exists(ctx.run_id, artifacts.FIGURES):
        catalog = ctx.store.load(ctx.run_id, artifacts.FIGURES, FigureCatalog)
        figures_total = len(catalog.figures)
        figures_worthy = sum(1 for f in catalog.figures if f.question_worthy)
    elif ctx.store.exists(ctx.run_id, artifacts.DOCUMENT):
        figures_total = len(
            ctx.store.load(ctx.run_id, artifacts.DOCUMENT, StructuredDocument).figures
        )

    usage_records = ctx.store.read_jsonl(ctx.run_id, artifacts.USAGE_LOG, UsageRecord)
    by_stage, by_unit, by_item, total = summarize_usage(usage_records)
    models: dict[str, str] = {}
    for usage in usage_records:
        models.setdefault(usage.stage or "unknown", usage.model)
    timings = ctx.store.read_jsonl(ctx.run_id, artifacts.TIMINGS_LOG, StageTiming)

    return RunReport(
        run_id=ctx.run_id,
        source_path=meta.source_path,
        document_hash=meta.document_hash,
        total_questions_requested=meta.total_questions or 0,
        units_planned=units_planned,
        units_covered=len({i.model_metadata.unit_id for i in items}),
        sections_total=sections_total,
        sections_covered=len({i.source.section_id for i in delivered}),
        generated=len(items),
        verified=by_status.get(QuestionStatus.VERIFIED, 0),
        flagged=by_status.get(QuestionStatus.FLAGGED, 0),
        discarded=by_status.get(QuestionStatus.DISCARDED, 0),
        exported=meta.exported_count or 0,
        discard_reasons=dict(
            Counter(i.status.reason for i in items if i.status.status == QuestionStatus.DISCARDED)
        ),
        flag_reasons=dict(
            Counter(i.status.reason for i in items if i.status.status == QuestionStatus.FLAGGED)
        ),
        deterministic_failures=dict(deterministic),
        judge_verdicts=dict(judge_verdicts),
        duplicates=duplicates,
        styles_requested=styles_requested,
        styles_delivered=dict(Counter(i.style.value for i in delivered)),
        figures_total=figures_total,
        figures_question_worthy=figures_worthy,
        image_questions=sum(1 for i in items if i.image is not None),
        stage_timings=timings,
        usage_by_stage=by_stage,
        usage_by_unit=by_unit,
        usage_by_item=by_item,
        usage_total=total,
        models=models,
    )


class ReportStage:
    def run(self, ctx: RunContext) -> RunReport:
        report = build_run_report(ctx)
        ctx.store.save(ctx.run_id, artifacts.RUN_REPORT, report)
        ctx.log(
            "run report refreshed",
            verified=report.verified,
            flagged=report.flagged,
            discarded=report.discarded,
            cost_usd=round(report.usage_total.cost_usd, 4),
        )
        return report
