from __future__ import annotations

from medquiz_generator.domain.models import (
    QuestionStatus,
    RunReport,
    StyleScores,
    Taxonomy,
    TaxonomySubject,
    TaxonomyTopic,
    UsageSummary,
)
from tests.unit.test_store_and_usage import make_item


def test_quiz_item_helpers() -> None:
    item = make_item()
    assert item.correct_option is not None and item.correct_option.id == "A"
    flagged = item.with_status(QuestionStatus.FLAGGED, "near duplicate", duplicate_of="q0")
    assert flagged.status.status == QuestionStatus.FLAGGED
    assert flagged.status.duplicate_of == "q0"
    assert item.status.status == QuestionStatus.GENERATED  # immutable update


def test_taxonomy_lookup() -> None:
    taxonomy = Taxonomy(
        run_id="r",
        document_hash="h",
        subjects=[
            TaxonomySubject(
                name="Cardiologia",
                topics=[TaxonomyTopic(name="Ipertensione", section_ids=["s1", "s2"])],
            )
        ],
    )
    label = taxonomy.label_for("s2")
    assert label is not None and (label.subject, label.topic) == ("Cardiologia", "Ipertensione")
    assert taxonomy.label_for("nope") is None
    assert taxonomy.section_count == 2


def test_style_scores_mean_and_dict() -> None:
    scores = StyleScores(
        ssm_style=5, distractor_quality=3, explanation_clarity=4, groundedness=4, rationale="ok"
    )
    assert scores.mean == 4.0
    assert scores.as_dict()["distractor_quality"] == 3


def test_run_report_cost_per_delivered_question() -> None:
    report = RunReport(
        run_id="r", verified=8, flagged=2, discarded=5, usage_total=UsageSummary(cost_usd=1.0)
    )
    assert report.cost_per_delivered_question_usd == 0.1
    assert RunReport(run_id="r").cost_per_delivered_question_usd is None
