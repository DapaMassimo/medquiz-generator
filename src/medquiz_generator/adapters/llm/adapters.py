"""LLM adapters implementing the generator, judge, taxonomy-builder and style-judge ports."""

from __future__ import annotations

import logging
import re

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

# Answer markers the model may copy from few-shot examples into its own option texts.
_ANSWER_MARKER = re.compile(
    r"\s*(\[corretta\]|\((?:risposta\s+)?corretta\)|✅|✔|✓)\s*$", re.IGNORECASE
)


def clean_option_text(text: str) -> str:
    return _ANSWER_MARKER.sub("", text).strip()


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
                options=[Option(id=o.id, text=clean_option_text(o.text)) for o in q.options],
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
        # Group per-section assignments into subjects → topics, preserving first-appearance
        # order; a section id assigned twice keeps its first assignment.
        subjects: dict[str, TaxonomySubject] = {}
        seen: set[str] = set()
        for a in wire.assignments:
            section_id = a.section_id.strip()
            if not section_id or section_id in seen:
                continue
            seen.add(section_id)
            subject_name = a.subject.strip() or "Senza materia"
            topic_name = a.topic.strip() or "Senza argomento"
            subject = subjects.setdefault(subject_name, TaxonomySubject(name=subject_name))
            topic = next((t for t in subject.topics if t.name == topic_name), None)
            if topic is None:
                topic = TaxonomyTopic(name=topic_name)
                subject.topics.append(topic)
            topic.section_ids.append(section_id)
        return Taxonomy(
            run_id=run_id, document_hash=document.content_hash, subjects=list(subjects.values())
        )


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
