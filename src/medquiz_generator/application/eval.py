"""Stage 8 — eval: rubric scoring of a run against the golden sets (opt-in harness).

* Style golden set: the real SSM questions in ``medquiz_generator/resources/ssm/ssm_examples.txt``.
* Regression golden set: items approved by a human and promoted from the review UI.
The style judge is a separate LLM call with its own prompt (not the verify judge).
"""

from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.report import latest_items
from medquiz_generator.domain.models import (
    EVAL_DIMENSIONS,
    EvalComparison,
    EvalItemResult,
    EvalReport,
    QuestionStatus,
    QuizItem,
    SsmExample,
    Stage,
    StructuredDocument,
)
from medquiz_generator.domain.ports import GoldenStore, StyleJudge
from medquiz_generator.domain.text import excerpt_around


def averages_of(results: list[EvalItemResult]) -> dict[str, float]:
    if not results:
        return {dim: 0.0 for dim in (*EVAL_DIMENSIONS, "mean")}
    averages = {
        dim: sum(r.scores.as_dict()[dim] for r in results) / len(results) for dim in EVAL_DIMENSIONS
    }
    averages["mean"] = sum(r.scores.mean for r in results) / len(results)
    return averages


def compare(a: EvalReport, b: EvalReport) -> EvalComparison:
    keys = (*EVAL_DIMENSIONS, "mean")
    return EvalComparison(
        run_a=a.run_id,
        run_b=b.run_id,
        averages_a={k: a.averages.get(k, 0.0) for k in keys},
        averages_b={k: b.averages.get(k, 0.0) for k in keys},
        delta={k: round(b.averages.get(k, 0.0) - a.averages.get(k, 0.0), 3) for k in keys},
    )


class EvalStage:
    def __init__(
        self,
        judge: StyleJudge,
        golden_style: list[SsmExample],
        golden_store: GoldenStore,
        *,
        include_flagged: bool = False,
    ) -> None:
        self._judge = judge
        self._golden_style = golden_style
        self._golden_store = golden_store
        self._include_flagged = include_flagged

    def candidates(self, ctx: RunContext) -> list[QuizItem]:
        allowed = {QuestionStatus.VERIFIED}
        if self._include_flagged:
            allowed.add(QuestionStatus.FLAGGED)
        return [i for i in latest_items(ctx) if i.status.status in allowed]

    def run(self, ctx: RunContext, *, limit: int | None = None) -> EvalReport:
        with ctx.stage(Stage.EVAL):
            regression = self._golden_store.load_all()
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            items = self.candidates(ctx)[:limit]
            results: list[EvalItemResult] = []
            for item in items:
                ctx.tracker.set_context(unit_id=item.model_metadata.unit_id, item_id=item.id)
                # Same bounded context the verify judge sees: groundedness must be judged on
                # the passage, not on the quote alone.
                section = document.section_by_id(item.source.section_id)
                passage = excerpt_around(section.text if section else "", item.source.quote)
                scores = self._judge.score(item, self._golden_style, regression, passage)
                results.append(EvalItemResult(item_id=item.id, scores=scores))
            ctx.tracker.set_context()
            report = EvalReport(
                run_id=ctx.run_id,
                judge_model=self._judge.model_name,
                golden_style_count=len(self._golden_style),
                golden_regression_count=len(regression),
                items=results,
                averages=averages_of(results),
            )
            ctx.store.save(ctx.run_id, artifacts.EVAL_REPORT, report)
            ctx.log("eval finished", items=len(results), mean=round(report.averages["mean"], 2))
        return report
