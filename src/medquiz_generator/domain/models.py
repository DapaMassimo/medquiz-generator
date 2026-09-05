"""Domain entities (Pydantic v2). Framework-free: no LangChain, Docling or Streamlit here.

Every artefact persisted by a pipeline stage is one of these models, so the on-disk JSON is
always the canonical schema and stages can be re-run independently from their inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Stage(StrEnum):
    PARSE = "parse"
    PLAN = "plan"
    CLASSIFY = "classify"
    GENERATE = "generate"
    VERIFY = "verify"
    DEDUP = "dedup"
    EXPORT = "export"
    EVAL = "eval"


class RunMeta(BaseModel):
    """Small header persisted per run: what was processed and with which parameters."""

    run_id: str
    source_path: str = ""
    document_hash: str = ""
    parser: str = ""
    total_questions: int | None = None
    exported_count: int | None = None
    exported_files: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- parse


class Section(BaseModel):
    """A node of the document outline with its own body text (children excluded)."""

    id: str
    title: str
    level: int = Field(ge=0, description="0 = document root/title, 1 = chapter, 2 = section…")
    path: list[str] = Field(default_factory=list, description="Titles from the root to here.")
    parent_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = ""

    @property
    def path_str(self) -> str:
        return " > ".join(self.path) if self.path else self.title

    @property
    def char_count(self) -> int:
        return len(self.text)


class StructuredDocument(BaseModel):
    """Output of the *parse* stage: a flat, ordered list of sections with page ranges."""

    source_path: str
    content_hash: str
    title: str
    page_count: int = Field(ge=1)
    sections: list[Section]
    parser: str = "unknown"
    parsed_at: datetime = Field(default_factory=utcnow)

    def section_by_id(self, section_id: str) -> Section | None:
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    @property
    def total_chars(self) -> int:
        return sum(s.char_count for s in self.sections)


# ---------------------------------------------------------------------------- plan


class PlanUnit(BaseModel):
    """One generation unit: a section, or a group of small sibling subsections."""

    id: str
    title: str
    section_ids: list[str]
    page_start: int
    page_end: int
    char_count: int
    weight: float = Field(ge=0.0, description="Share of the document's text (0..1).")
    questions: int = Field(ge=0, description="Questions budgeted for this unit.")
    note: str = ""


class GenerationPlan(BaseModel):
    """Output of the *plan* stage. Hand-editable before generation."""

    run_id: str
    document_hash: str
    total_questions: int = Field(ge=1)
    min_unit_chars: int
    units: list[PlanUnit]
    skipped_section_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def planned_questions(self) -> int:
        return sum(u.questions for u in self.units)


# ------------------------------------------------------------------------ taxonomy


class TaxonomyTopic(BaseModel):
    name: str
    section_ids: list[str] = Field(default_factory=list)


class TaxonomySubject(BaseModel):
    name: str
    topics: list[TaxonomyTopic] = Field(default_factory=list)


class SectionLabel(BaseModel):
    subject: str
    topic: str


class Taxonomy(BaseModel):
    """Output of the *classify* stage: subjects → topics → sections, extracted from the text."""

    run_id: str
    document_hash: str
    source: Literal["extracted", "provided"] = "extracted"
    subjects: list[TaxonomySubject]
    created_at: datetime = Field(default_factory=utcnow)

    def label_for(self, section_id: str) -> SectionLabel | None:
        for subject in self.subjects:
            for topic in subject.topics:
                if section_id in topic.section_ids:
                    return SectionLabel(subject=subject.name, topic=topic.name)
        return None

    @property
    def section_count(self) -> int:
        return sum(len(t.section_ids) for s in self.subjects for t in s.topics)


# ------------------------------------------------------------------------ questions

OptionId = Literal["A", "B", "C", "D", "E"]
OPTION_IDS: tuple[OptionId, ...] = ("A", "B", "C", "D", "E")


class Difficulty(StrEnum):
    FACILE = "facile"
    MEDIA = "media"
    DIFFICILE = "difficile"


class Option(BaseModel):
    id: OptionId
    text: str


class SourceRef(BaseModel):
    section_id: str
    section_title: str
    section_path: str
    page_start: int
    page_end: int
    page: int | None = Field(default=None, description="Page the quote comes from, if known.")
    quote: str


class QuestionStatus(StrEnum):
    GENERATED = "generated"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    DISCARDED = "discarded"


class StatusInfo(BaseModel):
    status: QuestionStatus = QuestionStatus.GENERATED
    reason: str = ""
    checks: list[str] = Field(default_factory=list, description="Deterministic checks failed.")
    judge_verdict: str | None = None
    judge_reason: str | None = None
    duplicate_of: str | None = None


class ModelMetadata(BaseModel):
    model: str
    run_id: str
    unit_id: str = ""
    generated_at: datetime = Field(default_factory=utcnow)


class QuizItem(BaseModel):
    """The deliverable: one SSM-style multiple-choice question with source and status."""

    id: str
    materia: str
    argomento: str
    question: str
    options: list[Option]
    correct_option_id: OptionId
    explanation: str
    difficulty: Difficulty
    source: SourceRef
    status: StatusInfo = Field(default_factory=StatusInfo)
    model_metadata: ModelMetadata

    @property
    def correct_option(self) -> Option | None:
        for option in self.options:
            if option.id == self.correct_option_id:
                return option
        return None

    def with_status(self, status: QuestionStatus, reason: str = "", **extra: object) -> QuizItem:
        info = self.status.model_copy(update={"status": status, "reason": reason, **extra})
        return self.model_copy(update={"status": info})


class GeneratedQuestion(BaseModel):
    """Raw structured output of the generator, before it is stamped into a QuizItem."""

    question: str
    options: list[Option]
    correct_option_id: OptionId
    explanation: str
    difficulty: Difficulty
    quote: str = Field(description="Verbatim passage from the section supporting the answer.")
    page: int | None = None


class GeneratedBatch(BaseModel):
    questions: list[GeneratedQuestion]


class SsmExample(BaseModel):
    """A real exam question parsed from examples/ssm/ssm_examples.txt (golden style set)."""

    number: int
    question: str
    options: list[Option]
    correct_option_id: OptionId | None
    commentary: str = ""


# --------------------------------------------------------------------------- verify


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "discard", "flag"]
    reason: str
    answer_supported: bool = True
    single_correct: bool = True
    distractors_wrong: bool = True


class VerificationRecord(BaseModel):
    item_id: str
    deterministic_failures: list[str]
    judge: JudgeVerdict | None
    final_status: QuestionStatus
    reason: str


class VerificationReport(BaseModel):
    run_id: str
    records: list[VerificationRecord]
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------- dedup


class DuplicatePair(BaseModel):
    item_id: str
    duplicate_of: str
    similarity: float


class DedupReport(BaseModel):
    run_id: str
    threshold: float
    pairs: list[DuplicatePair]
    created_at: datetime = Field(default_factory=utcnow)


# ----------------------------------------------------------------------------- eval

EVAL_DIMENSIONS: tuple[str, ...] = (
    "ssm_style",
    "distractor_quality",
    "explanation_clarity",
    "groundedness",
)


class StyleScores(BaseModel):
    """Rubric used by the eval judge (1-5 per dimension) with a rationale."""

    ssm_style: int = Field(ge=1, le=5)
    distractor_quality: int = Field(ge=1, le=5)
    explanation_clarity: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    rationale: str

    DIMENSIONS: ClassVar[tuple[str, ...]] = EVAL_DIMENSIONS

    def as_dict(self) -> dict[str, int]:
        return {dim: int(getattr(self, dim)) for dim in EVAL_DIMENSIONS}

    @property
    def mean(self) -> float:
        return sum(self.as_dict().values()) / len(EVAL_DIMENSIONS)


class EvalItemResult(BaseModel):
    item_id: str
    scores: StyleScores


class EvalReport(BaseModel):
    run_id: str
    judge_model: str
    golden_style_count: int
    golden_regression_count: int
    items: list[EvalItemResult]
    averages: dict[str, float]
    created_at: datetime = Field(default_factory=utcnow)


class EvalComparison(BaseModel):
    run_a: str
    run_b: str
    averages_a: dict[str, float]
    averages_b: dict[str, float]
    delta: dict[str, float]


# -------------------------------------------------------------------------- review


class ReviewDecision(BaseModel):
    item_id: str
    decision: Literal["approved", "rejected"]
    note: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class ReviewState(BaseModel):
    run_id: str
    decisions: dict[str, ReviewDecision] = Field(default_factory=dict)


class GoldenItem(BaseModel):
    """A human-approved item promoted to the regression golden set."""

    run_id: str
    promoted_at: datetime = Field(default_factory=utcnow)
    item: QuizItem


# --------------------------------------------------------------- observability


class UsageRecord(BaseModel):
    """One LLM/embedding call: what was called, how many tokens, and what it cost."""

    run_id: str
    stage: str
    model: str
    kind: Literal["chat", "embedding"] = "chat"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    unit_id: str = ""
    item_id: str = ""
    at: datetime = Field(default_factory=utcnow)


class StageTiming(BaseModel):
    stage: str
    duration_s: float
    started_at: datetime
    finished_at: datetime
    status: Literal["ok", "failed"] = "ok"


class UsageSummary(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, record: UsageRecord) -> None:
        self.calls += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cost_usd += record.cost_usd


class RunReport(BaseModel):
    """Persisted at the end of the pipeline (and refreshed after each stage)."""

    run_id: str
    source_path: str = ""
    document_hash: str = ""
    total_questions_requested: int = 0
    units_planned: int = 0
    units_covered: int = 0
    sections_total: int = 0
    sections_covered: int = 0
    generated: int = 0
    verified: int = 0
    flagged: int = 0
    discarded: int = 0
    exported: int = 0
    discard_reasons: dict[str, int] = Field(default_factory=dict)
    flag_reasons: dict[str, int] = Field(default_factory=dict)
    deterministic_failures: dict[str, int] = Field(default_factory=dict)
    judge_verdicts: dict[str, int] = Field(default_factory=dict)
    duplicates: int = 0
    stage_timings: list[StageTiming] = Field(default_factory=list)
    usage_by_stage: dict[str, UsageSummary] = Field(default_factory=dict)
    usage_by_unit: dict[str, UsageSummary] = Field(default_factory=dict)
    usage_by_item: dict[str, UsageSummary] = Field(default_factory=dict)
    usage_total: UsageSummary = Field(default_factory=UsageSummary)
    cost_per_delivered_question_usd: float | None = None
    models: dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _compute_cost_per_question(self) -> RunReport:
        delivered = self.verified + self.flagged
        if delivered > 0:
            self.cost_per_delivered_question_usd = self.usage_total.cost_usd / delivered
        return self

    @property
    def total_duration_s(self) -> float:
        return sum(t.duration_s for t in self.stage_timings)
