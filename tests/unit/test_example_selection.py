from __future__ import annotations

import pytest

from medquiz_generator.application import artifacts
from medquiz_generator.application.classify import ClassifyStage
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.example_selection import (
    FirstNExampleSelector,
    SemanticExampleSelector,
    example_text,
    is_clinical_vignette,
)
from medquiz_generator.application.generate import GenerateStage
from medquiz_generator.application.plan import PlanStage
from medquiz_generator.config.settings import Settings
from medquiz_generator.domain.models import OPTION_IDS, Option, SsmExample, StructuredDocument
from tests.fakes.fake_llm import FakeEmbeddings, FakeGenerator, FakeTaxonomyBuilder


def example(number: int, question: str, correct: str = "risposta") -> SsmExample:
    return SsmExample(
        number=number,
        question=question,
        options=[
            Option(id=oid, text=correct if oid == "A" else f"altro {oid}") for oid in OPTION_IDS
        ],
        correct_option_id="A",
    )


POOL = [
    example(
        1, "Un uomo di 50 anni si presenta con disfagia paradossa e calo ponderale.", "acalasia"
    ),
    example(
        2, "Quale valore pressorio definisce l'ipertensione arteriosa di grado 1?", "140-159/90-99"
    ),
    example(3, "Una donna di 55 anni con dolori ossei diffusi e poliuria.", "paratiroidectomia"),
    example(4, "La glicemia a digiuno che definisce il diabete mellito è:", "126 mg/dL"),
    example(5, "Un paziente di 40 anni con indice di massa corporea 35 viene valutato.", "bypass"),
    example(
        6,
        "Il monitoraggio pressorio delle 24 ore è indicato nell'ipertensione da camice bianco.",
        "ABPM",
    ),
    example(
        7,
        "Quali sono le complicanze microvascolari del diabete?",
        "retinopatia nefropatia neuropatia",
    ),
]


def test_is_clinical_vignette() -> None:
    assert is_clinical_vignette("Un uomo di 50 anni si presenta in ambulatorio")
    assert is_clinical_vignette("Una donna di 55 anni riferisce")
    assert is_clinical_vignette("Il paziente lamenta dolore")
    assert is_clinical_vignette("Giunge in PS un bambino di 3 anni")
    assert not is_clinical_vignette("Quale valore pressorio definisce l'ipertensione?")
    assert not is_clinical_vignette("La classificazione di Berg suddivide i linfonodi in:")


def test_example_text_uses_correct_option() -> None:
    assert example_text(POOL[1]).endswith("140-159/90-99")


def test_first_n_selector() -> None:
    selector = FirstNExampleSelector(POOL)
    assert selector.strategy == "first"
    assert [e.number for e in selector.select("qualsiasi", 3)] == [1, 2, 3]
    assert selector.select("x", 0) == []


def test_semantic_selector_short_circuits_when_pool_fits() -> None:
    embeddings = FakeEmbeddings()
    selector = SemanticExampleSelector(embeddings, POOL[:3])
    assert selector.strategy == "semantic"
    assert [e.number for e in selector.select("ipertensione", 5)] == [1, 2, 3]
    assert embeddings.calls == 0  # no embedding spent when every example is used anyway
    assert selector.select("x", 0) == []


def test_semantic_selector_picks_closest_examples_and_embeds_pool_once() -> None:
    embeddings = FakeEmbeddings()
    selector = SemanticExampleSelector(
        embeddings, POOL, min_similarity=0.0, ensure_both_registers=False
    )
    section = (
        "1.2 Diagnosi\nL'ipertensione arteriosa si definisce con pressione sistolica uguale o "
        "superiore a 140 mmHg; il monitoraggio pressorio delle 24 ore è indicato nel sospetto "
        "di ipertensione da camice bianco."
    )
    chosen = selector.select(section, 3)
    numbers = {e.number for e in chosen}
    assert {2, 6} <= numbers  # the two hypertension examples share the most vocabulary
    assert embeddings.calls == 2  # pool once + query once
    selector.select("diabete mellito glicemia a digiuno", 3)
    assert embeddings.calls == 3  # only the new query is embedded
    diabetic = {e.number for e in selector.select("diabete mellito glicemia a digiuno", 2)}
    assert 4 in diabetic


def test_semantic_selector_falls_back_to_first_n_below_threshold() -> None:
    embeddings = FakeEmbeddings()
    # Threshold impossible to reach: every slot comes from the file order (FirstN behaviour).
    selector = SemanticExampleSelector(embeddings, POOL, min_similarity=0.99)
    chosen = selector.select("ipertensione arteriosa monitoraggio pressorio", 3)
    assert [e.number for e in chosen] == [1, 2, 3]
    assert selector.last_fallback_count == 3
    assert embeddings.calls == 2  # the query was still embedded (pool + query)


def test_semantic_selector_partial_fallback_keeps_convincing_matches_first() -> None:
    embeddings = FakeEmbeddings()
    selector = SemanticExampleSelector(
        embeddings, POOL, min_similarity=0.35, ensure_both_registers=False
    )
    # Only the examples sharing many words with the query can pass a strict threshold.
    chosen = selector.select(
        "Il monitoraggio pressorio delle 24 ore è indicato nell'ipertensione da camice bianco.", 3
    )
    numbers = [e.number for e in chosen]
    assert numbers[0] == 6  # convincing match first
    assert len(numbers) == 3 and len(set(numbers)) == 3
    assert 1 <= selector.last_fallback_count <= 2
    # Fallback slots are the first examples of the file not already chosen.
    fallback = numbers[3 - selector.last_fallback_count :]
    expected = [
        e.number for e in POOL if e.number not in numbers[: 3 - selector.last_fallback_count]
    ]
    assert fallback == expected[: selector.last_fallback_count]


def test_semantic_selector_guarantees_both_registers() -> None:
    embeddings = FakeEmbeddings()
    selector = SemanticExampleSelector(embeddings, POOL, min_similarity=0.0)
    # A purely factual query would rank the three factual hypertension/diabetes examples first;
    # the diversity rule swaps the last slot for the best clinical vignette.
    chosen = selector.select("ipertensione arteriosa valore pressorio grado glicemia diabete", 3)
    kinds = {is_clinical_vignette(e.question) for e in chosen}
    assert kinds == {True, False}
    assert len(chosen) == 3


def test_generate_stage_uses_selector_per_unit(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    PlanStage(min_unit_chars=400).run(ctx, 8)
    ClassifyStage(FakeTaxonomyBuilder()).run(ctx)
    generator = FakeGenerator()
    selector = SemanticExampleSelector(
        FakeEmbeddings(tracker=ctx.tracker), POOL, min_similarity=0.0
    )
    GenerateStage(generator, POOL, max_examples=3, selector=selector).run(ctx)
    assert all(len(seen) == 3 for seen in generator.examples_seen)
    assert len({tuple(seen) for seen in generator.examples_seen}) > 1  # varies with the section
    # Embedding usage is attributed to the generate stage.
    usage = [r for r in ctx.tracker.records() if r.kind == "embedding"]
    assert usage and all(r.stage == "generate" for r in usage)


def test_generate_stage_without_selector_takes_first_n(
    ctx: RunContext, fixture_document: StructuredDocument
) -> None:
    ctx.store.save(ctx.run_id, artifacts.DOCUMENT, fixture_document)
    PlanStage(min_unit_chars=400).run(ctx, 4)
    ClassifyStage(FakeTaxonomyBuilder()).run(ctx)
    generator = FakeGenerator()
    GenerateStage(generator, POOL, max_examples=4).run(ctx)
    assert all(seen == [1, 2, 3, 4] for seen in generator.examples_seen)


def test_settings_bound_max_examples() -> None:
    assert Settings(_env_file=None).max_examples_in_prompt == 5  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_examples_in_prompt=8)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_examples_in_prompt=2)  # type: ignore[call-arg]
