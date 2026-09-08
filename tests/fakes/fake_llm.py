"""Fake LLM-backed adapters honouring the domain ports. No network, deterministic."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from medquiz_generator.domain.models import (
    OPTION_IDS,
    Difficulty,
    FigureAssessment,
    FigureRef,
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
    UsageRecord,
)
from medquiz_generator.domain.ports import UsageTracker

_SENTENCE = re.compile(r"[^.!?]+[.!?]")


def first_sentence(text: str) -> str:
    match = _SENTENCE.search(text)
    return match.group(0).strip() if match else text[:120]


class FakeGenerator:
    """Emits ``n`` well-formed questions per unit, quoting the first sentence of the section."""

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        *,
        model_name: str = "fake-generator",
        mutate: Callable[[GeneratedQuestion, int], GeneratedQuestion] | None = None,
    ) -> None:
        self._tracker = tracker
        self._model_name = model_name
        self._mutate = mutate
        self.calls: list[tuple[str, int]] = []
        self.examples_seen: list[list[int]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        unit: PlanUnit,
        sections: list[Section],
        n: int,
        examples: list[SsmExample],
        label: SectionLabel | None,
    ) -> GeneratedBatch:
        self.calls.append((unit.id, n))
        self.examples_seen.append([e.number for e in examples])
        if self._tracker is not None:
            self._tracker.record(
                UsageRecord(
                    run_id="",
                    stage="",
                    model=self._model_name,
                    input_tokens=1000 + 10 * n,
                    output_tokens=200 * n,
                    duration_s=0.01,
                )
            )
        questions: list[GeneratedQuestion] = []
        for k in range(n):
            section = sections[k % len(sections)]
            question = GeneratedQuestion(
                question=(
                    f"Paziente valutato per {section.title}: "
                    f"quale affermazione è corretta? ({k + 1})"
                ),
                options=[
                    Option(id=oid, text=f"Opzione {oid} su {section.title} #{k + 1}")
                    for oid in OPTION_IDS
                ],
                correct_option_id="A",
                explanation=f"La risposta corretta deriva dal testo di {section.title}.",
                difficulty=Difficulty.MEDIA,
                quote=first_sentence(section.text),
                page=section.page_start,
                style=QuestionStyle.VIGNETTA if k % 2 == 0 else QuestionStyle.NOZIONISTICA,
            )
            if self._mutate is not None:
                question = self._mutate(question, k)
            questions.append(question)
        return GeneratedBatch(questions=questions)


class FakeJudge:
    def __init__(
        self,
        rule: Callable[[QuizItem, str], JudgeVerdict] | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._rule = rule
        self._tracker = tracker
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-judge"

    def judge(self, item: QuizItem, passage: str) -> JudgeVerdict:
        self.calls.append(item.id)
        if self._tracker is not None:
            self._tracker.record(
                UsageRecord(
                    run_id="", stage="", model="fake-judge", input_tokens=500, output_tokens=50
                )
            )
        if self._rule is not None:
            return self._rule(item, passage)
        return JudgeVerdict(verdict="pass", reason="supported by passage")


class FakeTaxonomyBuilder:
    """Subject = level-1 chapter title, topic = section title; every section is mapped."""

    @property
    def model_name(self) -> str:
        return "fake-taxonomy"

    def build(self, document: StructuredDocument, run_id: str) -> Taxonomy:
        subjects: dict[str, TaxonomySubject] = {}
        for section in document.sections:
            chapter = section.path[1] if len(section.path) > 1 else section.title
            subject = subjects.setdefault(chapter, TaxonomySubject(name=chapter))
            subject.topics.append(TaxonomyTopic(name=section.title, section_ids=[section.id]))
        return Taxonomy(
            run_id=run_id, document_hash=document.content_hash, subjects=list(subjects.values())
        )


class FakeEmbeddings:
    """Bag-of-words hashing embeddings: identical texts → identical vectors.

    Buckets come from blake2b, not ``hash()``: string hashing is salted per process, so
    ``hash()`` keeps that promise only within a single run. Across runs the tokens land in
    different buckets, the cosine ranking shifts, and any test asserting a specific order
    fails for some values of PYTHONHASHSEED and passes for others.
    """

    def __init__(self, dims: int = 64, tracker: UsageTracker | None = None) -> None:
        self._dims = dims
        self._tracker = tracker
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._tracker is not None:
            self._tracker.record(
                UsageRecord(
                    run_id="",
                    stage="",
                    model="fake-embeddings",
                    kind="embedding",
                    input_tokens=sum(len(t) // 4 for t in texts),
                )
            )
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dims
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % self._dims] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FakeFigureTriage:
    """Classifies by caption keyword: 'ECG'/'radiografia'/'TC' → clinical, else diagram."""

    def __init__(self, tracker: UsageTracker | None = None) -> None:
        self._tracker = tracker
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-vision"

    def classify(self, figure: FigureRef, image: bytes) -> FigureAssessment:
        self.calls.append(figure.id)
        if self._tracker is not None:
            self._tracker.record(
                UsageRecord(
                    run_id="", stage="", model="fake-vision", input_tokens=900, output_tokens=60
                )
            )
        clinical = any(k in figure.caption.lower() for k in ("ecg", "radiograf", "tc ", "rm "))
        return FigureAssessment(
            kind="clinical_image" if clinical else "diagram",
            description=f"figura {figure.id}",
            question_worthy=clinical,
            reason="fake triage",
        )


class FakeStyleJudge:
    def __init__(self, scorer: Callable[[QuizItem], StyleScores] | None = None) -> None:
        self._scorer = scorer
        self.calls: list[str] = []
        self.passages: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-style-judge"

    def score(
        self,
        item: QuizItem,
        golden_style: list[SsmExample],
        golden_regression: list[QuizItem],
        passage: str = "",
    ) -> StyleScores:
        self.calls.append(item.id)
        self.passages.append(passage)
        if self._scorer is not None:
            return self._scorer(item)
        return StyleScores(
            ssm_style=4,
            distractor_quality=4,
            explanation_clarity=5,
            groundedness=5,
            rationale="fake rubric",
        )
