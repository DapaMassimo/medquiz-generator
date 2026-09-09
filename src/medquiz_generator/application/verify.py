"""Stage 5 — verify: deterministic checks + independent LLM judge.

Deterministic checks never call a model and are the first anti-hallucination gate: a quote
that is not in the section text discards the question outright. The judge then sees only
the cited passage (plus a bounded context window) and the question — never the generator's
reasoning — and returns pass / discard / flag with a reason.
"""

from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    OPTION_IDS,
    JudgeVerdict,
    QuestionStatus,
    QuizItem,
    Stage,
    StructuredDocument,
    VerificationRecord,
    VerificationReport,
)
from medquiz_generator.domain.ports import QuestionJudge
from medquiz_generator.domain.text import contains_normalized, excerpt_around, normalize_for_match

# Failures that make a question unusable: discarded without asking the judge.
HARD_FAILURES = frozenset(
    {
        "question_empty",
        "options_count",
        "options_empty",
        "options_duplicate",
        "options_ids",
        "correct_option_invalid",
        "quote_missing",
        "quote_not_in_section",
        "section_unknown",
    }
)
# Failures worth a human look but not fatal on their own.
SOFT_FAILURES = frozenset({"page_out_of_range", "explanation_short", "quote_too_long"})

MAX_QUOTE_WORDS = 120


def deterministic_checks(item: QuizItem, document: StructuredDocument) -> list[str]:
    failures: list[str] = []
    if not item.question.strip():
        failures.append("question_empty")
    if len(item.options) != 5:
        failures.append("options_count")
    if any(not o.text.strip() for o in item.options):
        failures.append("options_empty")
    if [o.id for o in item.options] != list(OPTION_IDS):
        failures.append("options_ids")
    normalized = [normalize_for_match(o.text) for o in item.options if o.text.strip()]
    if len(set(normalized)) != len(normalized):
        failures.append("options_duplicate")
    if item.correct_option_id not in {o.id for o in item.options}:
        failures.append("correct_option_invalid")
    if len(item.explanation.split()) < 8:
        failures.append("explanation_short")

    section = document.section_by_id(item.source.section_id)
    if section is None:
        failures.append("section_unknown")
        return failures
    quote = item.source.quote.strip()
    if not quote:
        failures.append("quote_missing")
    elif not contains_normalized(section.text, quote):
        failures.append("quote_not_in_section")
    if len(quote.split()) > MAX_QUOTE_WORDS:
        failures.append("quote_too_long")
    page = item.source.page
    # A figure may be printed on a page outside the range of the section discussing it.
    figure_page = item.image.page if item.image is not None else None
    if (
        page is not None
        and page != figure_page
        and not (section.page_start <= page <= section.page_end)
    ):
        failures.append("page_out_of_range")
    return failures


def combine(failures: list[str], judge: JudgeVerdict | None) -> tuple[QuestionStatus, str]:
    hard = [f for f in failures if f in HARD_FAILURES]
    soft = [f for f in failures if f not in HARD_FAILURES]
    if hard:
        return QuestionStatus.DISCARDED, "deterministic: " + ", ".join(hard)
    if judge is not None:
        if judge.verdict == "discard":
            return QuestionStatus.DISCARDED, f"judge: {judge.reason}"
        if judge.verdict == "flag":
            return QuestionStatus.FLAGGED, f"judge: {judge.reason}"
    if soft:
        return QuestionStatus.FLAGGED, "deterministic: " + ", ".join(soft)
    return QuestionStatus.VERIFIED, "passed deterministic checks" + (
        " and judge" if judge is not None else " (judge disabled)"
    )


class VerifyStage:
    def __init__(
        self, judge: QuestionJudge | None, *, image_questions_require_review: bool = True
    ) -> None:
        self._judge = judge
        self._image_review = image_questions_require_review

    def verify_item(
        self, item: QuizItem, document: StructuredDocument, image: bytes | None = None
    ) -> VerificationRecord:
        failures = deterministic_checks(item, document)
        verdict: JudgeVerdict | None = None
        if self._judge is not None and not any(f in HARD_FAILURES for f in failures):
            section = document.section_by_id(item.source.section_id)
            passage = excerpt_around(section.text if section else "", item.source.quote)
            verdict = self._judge.judge(item, passage, image)
        status, reason = combine(failures, verdict)
        if item.image is not None and self._image_review and status == QuestionStatus.VERIFIED:
            # Visual claims cannot be checked against the text: a human must confirm them.
            status = QuestionStatus.FLAGGED
            reason = "image question: human review required; " + reason
        return VerificationRecord(
            item_id=item.id,
            deterministic_failures=failures,
            judge=verdict,
            final_status=status,
            reason=reason,
        )

    def run(self, ctx: RunContext) -> list[QuizItem]:
        with ctx.stage(Stage.VERIFY):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            generated = ctx.require_list(artifacts.QUESTIONS_GENERATED, QuizItem)
            previous: dict[str, QuizItem] = {}
            if ctx.store.exists(ctx.run_id, artifacts.QUESTIONS_VERIFIED):
                previous = {
                    i.id: i
                    for i in ctx.store.load_list(ctx.run_id, artifacts.QUESTIONS_VERIFIED, QuizItem)
                }
            # Records of items verified by an earlier (interrupted) invocation are kept, so the
            # report always covers every item, not just the ones judged this time.
            previous_records: dict[str, VerificationRecord] = {}
            if ctx.store.exists(ctx.run_id, artifacts.VERIFICATION_REPORT):
                previous_records = {
                    r.item_id: r
                    for r in ctx.store.load(
                        ctx.run_id, artifacts.VERIFICATION_REPORT, VerificationReport
                    ).records
                }
            records: list[VerificationRecord] = []
            verified: list[QuizItem] = []
            for item in generated:
                if item.id in previous and previous[item.id].status.status != "generated":
                    verified.append(previous[item.id])
                    if item.id in previous_records:
                        records.append(previous_records[item.id])
                    continue
                ctx.tracker.set_context(unit_id=item.model_metadata.unit_id, item_id=item.id)
                image: bytes | None = None
                if item.image is not None:
                    image_path = ctx.store.run_dir(ctx.run_id) / item.image.path
                    image = image_path.read_bytes() if image_path.exists() else None
                record = self.verify_item(item, document, image)
                records.append(record)
                verified.append(
                    item.with_status(
                        record.final_status,
                        record.reason,
                        checks=record.deterministic_failures,
                        judge_verdict=record.judge.verdict if record.judge else None,
                        judge_reason=record.judge.reason if record.judge else None,
                    )
                )
                ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_VERIFIED, list(verified))
            ctx.tracker.set_context()
            ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_VERIFIED, list(verified))
            ctx.store.save(
                ctx.run_id,
                artifacts.VERIFICATION_REPORT,
                VerificationReport(run_id=ctx.run_id, records=records),
            )
            counts = {
                status.value: sum(1 for i in verified if i.status.status == status)
                for status in QuestionStatus
            }
            ctx.log("verification finished", **counts)
        return verified
