from __future__ import annotations

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.dedup import DedupStage, cosine, find_duplicates
from medquiz_generator.domain.models import DedupReport, QuestionStatus, QuizItem, UsageRecord
from tests.fakes.fake_llm import FakeEmbeddings
from tests.unit.test_verify import good_item


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_find_duplicates_reports_later_item_once() -> None:
    vectors = [[1.0, 0.0], [0.99, 0.1], [0.98, 0.15], [0.0, 1.0]]
    pairs = find_duplicates(["a", "b", "c", "d"], vectors, threshold=0.95)
    assert [(p.item_id, p.duplicate_of) for p in pairs] == [("b", "a"), ("c", "a")]
    assert all(p.similarity >= 0.95 for p in pairs)


def test_dedup_stage_flags_near_duplicates_and_skips_discarded(ctx: RunContext) -> None:
    original = good_item("q1").with_status(QuestionStatus.VERIFIED, "ok")
    duplicate = good_item("q2").with_status(QuestionStatus.VERIFIED, "ok")
    different = good_item(
        "q3",
        question="Quale classe farmacologica è sconsigliata in associazione agli ACE-inibitori?",
        options=[o.model_copy(update={"text": f"altro {o.id}"}) for o in good_item().options],
    ).with_status(QuestionStatus.VERIFIED, "ok")
    discarded = good_item("q4").with_status(QuestionStatus.DISCARDED, "bad")
    ctx.store.save_list(
        ctx.run_id, artifacts.QUESTIONS_VERIFIED, [original, duplicate, different, discarded]
    )
    embeddings = FakeEmbeddings(tracker=ctx.tracker)
    deduped = DedupStage(embeddings, threshold=0.9).run(ctx)
    by_id = {i.id: i for i in deduped}
    assert by_id["q1"].status.status == QuestionStatus.VERIFIED
    assert by_id["q2"].status.status == QuestionStatus.FLAGGED
    assert by_id["q2"].status.duplicate_of == "q1"
    assert by_id["q3"].status.status == QuestionStatus.VERIFIED
    assert by_id["q4"].status.status == QuestionStatus.DISCARDED
    report = ctx.store.load(ctx.run_id, artifacts.DEDUP_REPORT, DedupReport)
    assert [(p.item_id, p.duplicate_of) for p in report.pairs] == [("q2", "q1")]
    assert embeddings.calls == 1
    usage = ctx.store.read_jsonl(ctx.run_id, artifacts.USAGE_LOG, UsageRecord)
    assert usage and usage[0].kind == "embedding" and usage[0].stage == "dedup"
    stored = ctx.store.load_list(ctx.run_id, artifacts.QUESTIONS_DEDUPED, QuizItem)
    assert len(stored) == 4


def test_dedup_stage_disabled_passes_items_through(ctx: RunContext) -> None:
    items = [good_item("q1").with_status(QuestionStatus.VERIFIED, "ok")]
    ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_VERIFIED, list(items))
    deduped = DedupStage(None).run(ctx)
    assert deduped[0].status.status == QuestionStatus.VERIFIED
    assert ctx.store.load(ctx.run_id, artifacts.DEDUP_REPORT, DedupReport).pairs == []
