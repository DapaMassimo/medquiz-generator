from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.verify import (
    VerifyStage,
    combine,
    deterministic_checks,
)
from medquiz_generator.domain.models import (
    Difficulty,
    JudgeVerdict,
    ModelMetadata,
    Option,
    QuestionStatus,
    QuizItem,
    SourceRef,
    StructuredDocument,
    VerificationReport,
)
from tests.fakes.fake_llm import FakeJudge

GOOD_QUOTE = "Il monitoraggio pressorio delle 24 ore (ABPM) è indicato in caso di sospetta ipertensione da camice bianco"  # noqa: E501


def good_item(item_id: str = "u001-q01", **overrides: object) -> QuizItem:
    base = QuizItem(
        id=item_id,
        materia="Cardiologia",
        argomento="Ipertensione",
        question="Un paziente con valori elevati solo in ambulatorio: quale esame è indicato?",
        options=[
            Option(id="A", text="Monitoraggio pressorio delle 24 ore"),
            Option(id="B", text="Ecocardiogramma"),
            Option(id="C", text="Test da sforzo"),
            Option(id="D", text="Radiografia del torace"),
            Option(id="E", text="Fundus oculi"),
        ],
        correct_option_id="A",
        explanation="Il monitoraggio delle 24 ore è indicato nel sospetto di ipertensione da camice bianco secondo il testo.",  # noqa: E501
        difficulty=Difficulty.MEDIA,
        source=SourceRef(
            section_id="s003",
            section_title="1.2 Diagnosi",
            section_path="Doc > Cap 1 > 1.2 Diagnosi",
            page_start=2,
            page_end=3,
            page=2,
            quote=GOOD_QUOTE,
        ),
        model_metadata=ModelMetadata(model="fake", run_id="run-test", unit_id="u001"),
    )
    return base.model_copy(update=overrides)


def test_good_item_passes_all_checks(fixture_document: StructuredDocument) -> None:
    assert deterministic_checks(good_item(), fixture_document) == []


def test_quote_with_layout_noise_still_matches(fixture_document: StructuredDocument) -> None:
    noisy = good_item(
        source=good_item().source.model_copy(
            update={"quote": "il  monitoraggio pressorio delle 24 ore (abpm) e' indicato"}
        )
    )
    # apostrophe vs accent is a wording change ("e'" ≠ "è") → must fail…
    assert "quote_not_in_section" in deterministic_checks(noisy, fixture_document)
    noisy_ok = good_item(
        source=good_item().source.model_copy(
            update={"quote": "  Il MONITORAGGIO pressorio,\n delle 24 ore (ABPM) è indicato "}
        )
    )
    # …while spacing/case/punctuation differences must not.
    assert deterministic_checks(noisy_ok, fixture_document) == []


def test_quote_absent_is_hard_failure(fixture_document: StructuredDocument) -> None:
    item = good_item(source=good_item().source.model_copy(update={"quote": "frase inventata"}))
    failures = deterministic_checks(item, fixture_document)
    assert failures == ["quote_not_in_section"]
    assert combine(failures, None) == (
        QuestionStatus.DISCARDED,
        "deterministic: quote_not_in_section",
    )


def test_empty_quote_is_hard_failure(fixture_document: StructuredDocument) -> None:
    item = good_item(source=good_item().source.model_copy(update={"quote": "   "}))
    assert "quote_missing" in deterministic_checks(item, fixture_document)


def test_two_identical_options_fail(fixture_document: StructuredDocument) -> None:
    options = good_item().options
    options[2] = Option(id="C", text="monitoraggio pressorio delle 24 ore.")
    failures = deterministic_checks(good_item(options=options), fixture_document)
    assert failures == ["options_duplicate"]


def test_wrong_option_count_and_ids(fixture_document: StructuredDocument) -> None:
    item = good_item(options=good_item().options[:4])
    failures = deterministic_checks(item, fixture_document)
    assert "options_count" in failures and "options_ids" in failures


def test_empty_option_and_invalid_correct(fixture_document: StructuredDocument) -> None:
    options = good_item().options
    options[4] = Option(id="E", text="  ")
    failures = deterministic_checks(good_item(options=options), fixture_document)
    assert "options_empty" in failures
    item = good_item(
        correct_option_id="E", options=[*good_item().options[:4], Option(id="D", text="dup id")]
    )
    failures = deterministic_checks(item, fixture_document)
    assert "correct_option_invalid" in failures


def test_page_out_of_range_is_soft(fixture_document: StructuredDocument) -> None:
    item = good_item(source=good_item().source.model_copy(update={"page": 9}))
    failures = deterministic_checks(item, fixture_document)
    assert failures == ["page_out_of_range"]
    status, reason = combine(failures, JudgeVerdict(verdict="pass", reason="ok"))
    assert status == QuestionStatus.FLAGGED and "page_out_of_range" in reason


def test_image_question_may_cite_the_figure_page_outside_the_section(
    fixture_document: StructuredDocument,
) -> None:
    from medquiz_generator.domain.models import ImageRef

    image = ImageRef(figure_id="f001", path="assets/figures/f001.png", page=9)
    item = good_item(image=image, source=good_item().source.model_copy(update={"page": 9}))
    assert deterministic_checks(item, fixture_document) == []
    other = good_item(image=image, source=good_item().source.model_copy(update={"page": 8}))
    assert deterministic_checks(other, fixture_document) == ["page_out_of_range"]


def test_unknown_section(fixture_document: StructuredDocument) -> None:
    item = good_item(source=good_item().source.model_copy(update={"section_id": "s999"}))
    assert deterministic_checks(item, fixture_document) == ["section_unknown"]


def test_combine_judge_verdicts() -> None:
    assert combine([], JudgeVerdict(verdict="pass", reason="ok"))[0] == QuestionStatus.VERIFIED
    assert combine([], JudgeVerdict(verdict="flag", reason="dubbio"))[0] == QuestionStatus.FLAGGED
    status, reason = combine([], JudgeVerdict(verdict="discard", reason="non supportata"))
    assert status == QuestionStatus.DISCARDED and reason == "judge: non supportata"
    status, reason = combine([], None)
    assert status == QuestionStatus.VERIFIED and "judge disabled" in reason


def test_verify_stage_applies_judge_and_persists(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    items = [
        good_item("q-pass"),
        good_item("q-flag"),
        good_item("q-judge-discard"),
        good_item(
            "q-det-discard",
            source=good_item().source.model_copy(update={"quote": "inventata"}),
        ),
    ]
    ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, list(items))

    def rule(item: QuizItem, passage: str) -> JudgeVerdict:
        assert GOOD_QUOTE.lower() in passage.lower()
        if item.id == "q-flag":
            return JudgeVerdict(verdict="flag", reason="ambigua")
        if item.id == "q-judge-discard":
            return JudgeVerdict(verdict="discard", reason="non supportata", answer_supported=False)
        return JudgeVerdict(verdict="pass", reason="ok")

    judge = FakeJudge(rule, tracker=ctx.tracker)
    verified = VerifyStage(judge).run(ctx)
    by_id = {i.id: i for i in verified}
    assert by_id["q-pass"].status.status == QuestionStatus.VERIFIED
    assert by_id["q-flag"].status.status == QuestionStatus.FLAGGED
    assert by_id["q-flag"].status.judge_reason == "ambigua"
    assert by_id["q-judge-discard"].status.status == QuestionStatus.DISCARDED
    assert by_id["q-det-discard"].status.status == QuestionStatus.DISCARDED
    assert by_id["q-det-discard"].status.checks == ["quote_not_in_section"]
    # The judge is never called for items that failed hard deterministic checks.
    assert "q-det-discard" not in judge.calls
    report = ctx.store.load(ctx.run_id, artifacts.VERIFICATION_REPORT, VerificationReport)
    assert len(report.records) == 4
    # Re-running is idempotent: previously verified items are not judged again, and the
    # report still carries one record per item.
    VerifyStage(judge).run(ctx)
    assert len(judge.calls) == 3
    report = ctx.store.load(ctx.run_id, artifacts.VERIFICATION_REPORT, VerificationReport)
    assert sorted(r.item_id for r in report.records) == sorted(i.id for i in items)


def test_verify_stage_without_judge(ctx: RunContext, fixture_document: StructuredDocument) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, [good_item("q1")])
    [item] = VerifyStage(None).run(ctx)
    assert item.status.status == QuestionStatus.VERIFIED
    assert item.status.judge_verdict is None
