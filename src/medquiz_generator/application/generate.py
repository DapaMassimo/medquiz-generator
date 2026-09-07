"""Stage 4 — generate: one structured LLM call per plan unit → QuizItems (status generated).

Resumable: units already present in ``questions.generated.json`` are skipped and the artefact
is re-saved after every unit, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import random
import re

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    OPTION_IDS,
    GeneratedQuestion,
    GenerationPlan,
    ModelMetadata,
    Option,
    PlanUnit,
    QuizItem,
    Section,
    SectionLabel,
    SourceRef,
    SsmExample,
    Stage,
    StructuredDocument,
    Taxonomy,
)
from medquiz_generator.domain.ports import ExampleSelector, QuestionGenerator
from medquiz_generator.domain.text import contains_normalized


def pick_source_section(question: GeneratedQuestion, sections: list[Section]) -> Section:
    """The section whose text contains the quote; falls back to the page, then the first."""
    for section in sections:
        if contains_normalized(section.text, question.quote):
            return section
    if question.page is not None:
        for section in sections:
            if section.page_start <= question.page <= section.page_end:
                return section
    return sections[0]


_LETTER_REF = re.compile(
    r"\b(risposta|risposte|opzione|opzioni|alternativa|alternative|lettera|la|le)\s+([A-E])\b"
)


def shuffle_options(question: GeneratedQuestion, seed: str) -> GeneratedQuestion:
    """Deterministically reorder the options and relabel them A-E.

    Models tend to put the correct answer in the same slot (gpt-4.1 wrote 22/30 answers as A);
    reordering in code removes the bias regardless of the model. Letter references in the
    explanation ("la risposta B è errata") are remapped to the new letters.
    """
    if len(question.options) != len(OPTION_IDS):
        return question
    order = list(range(len(question.options)))
    random.Random(seed).shuffle(order)
    old_to_new: dict[str, str] = {
        str(question.options[old].id): OPTION_IDS[new] for new, old in enumerate(order)
    }
    options = [
        Option(id=OPTION_IDS[new], text=question.options[old].text) for new, old in enumerate(order)
    ]
    explanation = _LETTER_REF.sub(
        lambda m: f"{m.group(1)} {old_to_new.get(m.group(2), m.group(2))}", question.explanation
    )
    return question.model_copy(
        update={
            "options": options,
            "correct_option_id": old_to_new[question.correct_option_id],
            "explanation": explanation,
        }
    )


def to_quiz_item(
    question: GeneratedQuestion,
    *,
    item_id: str,
    unit: PlanUnit,
    sections: list[Section],
    label: SectionLabel | None,
    model: str,
    run_id: str,
) -> QuizItem:
    section = pick_source_section(question, sections)
    question = shuffle_options(question, seed=f"{run_id}:{item_id}")
    return QuizItem(
        id=item_id,
        materia=label.subject if label else "",
        argomento=label.topic if label else section.title,
        question=question.question.strip(),
        options=question.options,
        correct_option_id=question.correct_option_id,
        explanation=question.explanation.strip(),
        difficulty=question.difficulty,
        source=SourceRef(
            section_id=section.id,
            section_title=section.title,
            section_path=section.path_str,
            page_start=section.page_start,
            page_end=section.page_end,
            page=question.page,
            quote=question.quote.strip(),
        ),
        model_metadata=ModelMetadata(model=model, run_id=run_id, unit_id=unit.id),
    )


class GenerateStage:
    def __init__(
        self,
        generator: QuestionGenerator,
        examples: list[SsmExample],
        *,
        max_examples: int = 5,
        selector: ExampleSelector | None = None,
    ) -> None:
        self._generator = generator
        self._examples = list(examples)
        self._max_examples = max_examples
        self._selector = selector

    def examples_for(self, sections: list[Section]) -> list[SsmExample]:
        """Few-shot examples for a unit: selector-driven when configured, else the first N."""
        if self._selector is None:
            return self._examples[: self._max_examples]
        query = "\n\n".join(f"{s.path_str}\n{s.text}" for s in sections)
        return self._selector.select(query, self._max_examples)

    def run(self, ctx: RunContext, *, unit_ids: list[str] | None = None) -> list[QuizItem]:
        with ctx.stage(Stage.GENERATE):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            plan = ctx.require(artifacts.PLAN, GenerationPlan)
            taxonomy = ctx.require(artifacts.TAXONOMY, Taxonomy)
            items: list[QuizItem] = (
                ctx.store.load_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, QuizItem)
                if ctx.store.exists(ctx.run_id, artifacts.QUESTIONS_GENERATED)
                else []
            )
            done_units = {item.model_metadata.unit_id for item in items}
            if not self._examples:
                ctx.warn("no SSM examples loaded: few-shot disabled")

            for unit in plan.units:
                if unit.questions <= 0 or unit.id in done_units:
                    continue
                if unit_ids is not None and unit.id not in unit_ids:
                    continue
                sections = [
                    s for sid in unit.section_ids if (s := document.section_by_id(sid)) is not None
                ]
                if not sections:
                    ctx.log("unit has no sections in document; skipped", unit_id=unit.id)
                    continue
                label = taxonomy.label_for(sections[0].id)
                ctx.tracker.set_context(unit_id=unit.id)
                examples = self.examples_for(sections)
                batch = self._generator.generate(unit, sections, unit.questions, examples, label)
                new_items = [
                    to_quiz_item(
                        question,
                        item_id=f"{unit.id}-q{index:02d}",
                        unit=unit,
                        sections=sections,
                        label=label,
                        model=self._generator.model_name,
                        run_id=ctx.run_id,
                    )
                    for index, question in enumerate(batch.questions, start=1)
                ]
                items.extend(new_items)
                ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, list(items))
                ctx.log(
                    "unit generated",
                    unit_id=unit.id,
                    requested=unit.questions,
                    produced=len(new_items),
                    examples=[e.number for e in examples],
                )
            ctx.tracker.set_context()
            ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, list(items))
            ctx.log("generation finished", items=len(items), model=self._generator.model_name)
        return items
