"""LLM adapters implementing the generator, judge, taxonomy-builder and style-judge ports."""

from __future__ import annotations

import logging

from medquiz_generator.adapters.llm import prompts
from medquiz_generator.adapters.llm.langchain_llm import StructuredLLM
from medquiz_generator.domain.models import (
    Difficulty,
    GeneratedBatch,
    GeneratedQuestion,
    JudgeVerdict,
    Option,
    PlanUnit,
    QuizItem,
    Section,
    SectionLabel,
    SsmExample,
    StructuredDocument,
    StyleScores,
    Taxonomy,
    TaxonomySubject,
    TaxonomyTopic,
)

log = logging.getLogger(__name__)


class LangChainQuestionGenerator:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def generate(
        self,
        unit: PlanUnit,
        sections: list[Section],
        n: int,
        examples: list[SsmExample],
        label: SectionLabel | None,
    ) -> GeneratedBatch:
        batch = self._llm.invoke(
            prompts.GENERATION_SYSTEM,
            prompts.generation_user_prompt(unit, sections, n, examples, label),
            prompts.WireBatch,
        )
        questions = [
            GeneratedQuestion(
                question=q.question,
                options=[Option(id=o.id, text=o.text) for o in q.options],
                correct_option_id=q.correct_option_id,
                explanation=q.explanation,
                difficulty=Difficulty(q.difficulty),
                quote=q.quote,
                page=q.page,
            )
            for q in batch.questions
        ]
        if len(questions) != n:
            log.warning(
                "Unit %s: requested %d questions, model returned %d", unit.id, n, len(questions)
            )
        return GeneratedBatch(questions=questions)


class LangChainQuestionJudge:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def judge(self, item: QuizItem, passage: str) -> JudgeVerdict:
        verdict = self._llm.invoke(
            prompts.JUDGE_SYSTEM, prompts.judge_user_prompt(item, passage), prompts.WireVerdict
        )
        return JudgeVerdict(
            verdict=verdict.verdict,
            reason=verdict.reason,
            answer_supported=verdict.answer_supported,
            single_correct=verdict.single_correct,
            distractors_wrong=verdict.distractors_wrong,
        )


class LangChainTaxonomyBuilder:
    """Extracts the taxonomy from the document outline (structure + LLM normalisation)."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def build(self, document: StructuredDocument, run_id: str) -> Taxonomy:
        wire = self._llm.invoke(
            prompts.TAXONOMY_SYSTEM, prompts.taxonomy_user_prompt(document), prompts.WireTaxonomy
        )
        subjects = [
            TaxonomySubject(
                name=s.name.strip(),
                topics=[
                    TaxonomyTopic(name=t.name.strip(), section_ids=list(t.section_ids))
                    for t in s.topics
                ],
            )
            for s in wire.subjects
        ]
        return Taxonomy(run_id=run_id, document_hash=document.content_hash, subjects=subjects)


class LangChainStyleJudge:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def score(
        self, item: QuizItem, golden_style: list[SsmExample], golden_regression: list[QuizItem]
    ) -> StyleScores:
        wire = self._llm.invoke(
            prompts.STYLE_JUDGE_SYSTEM,
            prompts.style_judge_user_prompt(item, golden_style, golden_regression),
            prompts.WireStyleScores,
        )
        return StyleScores(**wire.model_dump())
