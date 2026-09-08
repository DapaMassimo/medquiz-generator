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
    QuestionStyle,
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
                style=QuestionStyle(q.style),
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
        wire = self._llm.invoke(
            prompts.JUDGE_SYSTEM, prompts.judge_user_prompt(item, passage), prompts.WireVerdict
        )
        return derive_verdict(wire, item.correct_option_id)


def derive_verdict(wire: prompts.WireVerdict, marked: str) -> JudgeVerdict:
    """Turn per-option assessments into the verdict; the model's own verdict can only make it
    stricter (flag/discard), never lift a structural failure."""
    status: dict[str, str] = {str(a.id): str(a.status) for a in wire.options}
    supported = [oid for oid, st in status.items() if st == "supported"]
    answer_supported = status.get(marked) == "supported"
    single_correct = len(supported) <= 1
    distractors_wrong = all(oid == marked for oid in supported)
    verdict: str = wire.verdict
    reason = wire.reason
    if not answer_supported:
        verdict = "discard"
        reason = (
            f"la risposta marcata {marked} non è sostenuta dal passaggio "
            f"(stato: {status.get(marked, 'assente')}); {wire.reason}"
        )
    elif not distractors_wrong:
        others = ", ".join(o for o in supported if o != marked)
        verdict = "discard"
        reason = f"anche l'opzione {others} risulta sostenuta dal passaggio; {wire.reason}"
    elif verdict == "pass" and len(status) < 5:
        verdict = "flag"
        reason = f"valutazione incompleta delle opzioni ({len(status)}/5); {wire.reason}"
    explanation_supported = wire.explanation_status == "supported"
    if verdict == "pass" and not explanation_supported:
        # The answer stands, but the explanation brings in facts the reader cannot verify on
        # the source: a human decides whether to trim it.
        verdict = "flag"
        note = wire.explanation_note or "dettagli assenti dal testo"
        reason = (
            f"spiegazione non pienamente sostenuta dal passaggio "
            f"({wire.explanation_status}: {note}); {wire.reason}"
        )
    return JudgeVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason,
        answer_supported=answer_supported,
        single_correct=single_correct,
        distractors_wrong=distractors_wrong,
        option_status=status,
        explanation_supported=explanation_supported,
        explanation_note=wire.explanation_note,
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
        self,
        item: QuizItem,
        golden_style: list[SsmExample],
        golden_regression: list[QuizItem],
        passage: str,
    ) -> StyleScores:
        wire = self._llm.invoke(
            prompts.STYLE_JUDGE_SYSTEM,
            prompts.style_judge_user_prompt(item, golden_style, golden_regression, passage),
            prompts.WireStyleScores,
        )
        return StyleScores(**wire.model_dump())
