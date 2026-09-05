"""Stage 7 — export: verified (+ optionally flagged) questions → JSON + XLSX.

Discarded questions never reach an export. Flagged ones are included only on request and
keep their status/reason so the marking is visible in the file.
"""

from __future__ import annotations

from pathlib import Path

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import QuestionStatus, QuizItem, ReviewState, Stage
from medquiz_generator.domain.ports import QuizExporter


def select_deliverable(items: list[QuizItem], *, include_flagged: bool) -> list[QuizItem]:
    allowed = {QuestionStatus.VERIFIED}
    if include_flagged:
        allowed.add(QuestionStatus.FLAGGED)
    return [i for i in items if i.status.status in allowed]


def select_approved(items: list[QuizItem], review: ReviewState) -> list[QuizItem]:
    return [
        i
        for i in items
        if i.status.status != QuestionStatus.DISCARDED
        and (decision := review.decisions.get(i.id)) is not None
        and decision.decision == "approved"
    ]


def load_items_for_export(ctx: RunContext) -> list[QuizItem]:
    if ctx.store.exists(ctx.run_id, artifacts.QUESTIONS_DEDUPED):
        return ctx.store.load_list(ctx.run_id, artifacts.QUESTIONS_DEDUPED, QuizItem)
    return ctx.require_list(artifacts.QUESTIONS_VERIFIED, QuizItem)


class ExportStage:
    def __init__(self, exporters: list[QuizExporter], *, include_flagged: bool = False) -> None:
        self._exporters = exporters
        self._include_flagged = include_flagged

    def export_items(
        self, ctx: RunContext, items: list[QuizItem], *, basename: str
    ) -> dict[str, Path]:
        target_dir = ctx.store.run_dir(ctx.run_id) / artifacts.EXPORT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for exporter in self._exporters:
            target = target_dir / f"{basename}.{exporter.format_name}"
            written[exporter.format_name] = exporter.export(items, target)
        return written

    def run(
        self,
        ctx: RunContext,
        *,
        include_flagged: bool | None = None,
        approved_only: bool = False,
    ) -> dict[str, Path]:
        with ctx.stage(Stage.EXPORT):
            items = load_items_for_export(ctx)
            if approved_only:
                review = (
                    ctx.store.load(ctx.run_id, artifacts.REVIEW_STATE, ReviewState)
                    if ctx.store.exists(ctx.run_id, artifacts.REVIEW_STATE)
                    else ReviewState(run_id=ctx.run_id)
                )
                selected = select_approved(items, review)
                basename = "quiz.approved"
            else:
                flagged = self._include_flagged if include_flagged is None else include_flagged
                selected = select_deliverable(items, include_flagged=flagged)
                basename = "quiz"
            written = self.export_items(ctx, selected, basename=basename)
            if not approved_only:
                ctx.update_meta(
                    exported_count=len(selected),
                    exported_files={k: str(v) for k, v in written.items()},
                )
            ctx.log("export finished", items=len(selected), files=list(written))
        return written
