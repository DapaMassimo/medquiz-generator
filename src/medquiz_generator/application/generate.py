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
    FigureCatalog,
    FigureRef,
    GeneratedQuestion,
    GenerationPlan,
    ImageRef,
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
from medquiz_generator.domain.ports import (
    ExampleSelector,
    ImageQuestionGenerator,
    QuestionGenerator,
)
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
        style=question.style,
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
        image_generator: ImageQuestionGenerator | None = None,
    ) -> None:
        self._generator = generator
        self._examples = list(examples)
        self._max_examples = max_examples
        self._selector = selector
        self._image_generator = image_generator

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
            if self._image_generator is not None:
                items = self._generate_image_questions(ctx, document, taxonomy, items)
            ctx.tracker.set_context()
            ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, list(items))
            ctx.log("generation finished", items=len(items), model=self._generator.model_name)
        return items

    def _generate_image_questions(
        self,
        ctx: RunContext,
        document: StructuredDocument,
        taxonomy: Taxonomy,
        items: list[QuizItem],
    ) -> list[QuizItem]:
        """One question per question-worthy figure (figures.json), on top of the plan."""
        assert self._image_generator is not None
        if not ctx.store.exists(ctx.run_id, artifacts.FIGURES):
            return items
        catalog = ctx.store.load(ctx.run_id, artifacts.FIGURES, FigureCatalog)
        done = {i.image.figure_id for i in items if i.image is not None}
        run_dir = ctx.store.run_dir(ctx.run_id)
        for figure in catalog.figures:
            if not figure.question_worthy or figure.id in done:
                continue
            section = document.section_by_id(figure.section_id)
            image_path = run_dir / figure.path
            if section is None or not image_path.exists():
                ctx.warn("figure skipped: section or image missing", figure_id=figure.id)
                continue
            label = taxonomy.label_for(section.id)
            ctx.tracker.set_context(unit_id=figure.section_id, item_id=f"fig-{figure.id}")
            batch = self._image_generator.generate(
                figure, image_path.read_bytes(), section, self.examples_for([section]), label
            )
            for index, question in enumerate(batch.questions, start=1):
                item = to_quiz_item(
                    question,
                    item_id=f"fig-{figure.id}-q{index:02d}",
                    unit=_figure_unit(figure, section),
                    sections=[section],
                    label=label,
                    model=self._image_generator.model_name,
                    run_id=ctx.run_id,
                )
                items.append(
                    item.model_copy(
                        update={
                            "image": ImageRef(
                                figure_id=figure.id,
                                path=figure.path,
                                caption=figure.caption,
                                page=figure.page,
                            )
                        }
                    )
                )
            ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_GENERATED, list(items))
            ctx.log("figure question generated", figure_id=figure.id, produced=len(batch.questions))
        return items


def _figure_unit(figure: FigureRef, section: Section) -> PlanUnit:
    """A synthetic unit so image questions carry a unit id like the others."""
    return PlanUnit(
        id=f"fig-{figure.id}",
        title=figure.caption or section.title,
        section_ids=[section.id],
        page_start=figure.page,
        page_end=figure.page,
        char_count=section.char_count,
        weight=0.0,
        questions=1,
    )
