"""Ports (Protocols) that the application layer depends on. Adapters implement them.

Keeping every external capability behind a Protocol is what lets the whole pipeline run in
tests with in-memory fakes and no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from medquiz_generator.domain.models import (
    FigureAssessment,
    FigureRef,
    GeneratedBatch,
    JudgeVerdict,
    PlanUnit,
    QuizItem,
    Section,
    SectionLabel,
    SsmExample,
    StructuredDocument,
    StyleScores,
    Taxonomy,
    UsageRecord,
)

M = TypeVar("M", bound=BaseModel)


class DocumentParser(Protocol):
    """Turns a native PDF into a StructuredDocument (section tree with page ranges)."""

    @property
    def name(self) -> str: ...

    def parse(
        self, pdf_path: Path, *, use_cache: bool = True, assets_dir: Path | None = None
    ) -> StructuredDocument:
        """``assets_dir`` (when given) receives the figures as PNG files, listed in the
        document; without it figures are not extracted."""
        ...


class FigureTriage(Protocol):
    """Vision model classifying a figure and saying whether it can carry an exam question."""

    @property
    def model_name(self) -> str: ...

    def classify(self, figure: FigureRef, image: bytes) -> FigureAssessment: ...


class TaxonomyBuilder(Protocol):
    """Builds subjects → topics → sections from the document itself.

    A future ``ProvidedTaxonomyBuilder`` fed by a client file satisfies the same port.
    """

    @property
    def model_name(self) -> str: ...

    def build(self, document: StructuredDocument, run_id: str) -> Taxonomy: ...


class QuestionGenerator(Protocol):
    """Generates ``n`` questions for one plan unit from its section texts."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        unit: PlanUnit,
        sections: list[Section],
        n: int,
        examples: list[SsmExample],
        label: SectionLabel | None,
    ) -> GeneratedBatch: ...


class QuestionJudge(Protocol):
    """Independent verifier: sees only the quoted passage and the question, never the
    generator's reasoning."""

    @property
    def model_name(self) -> str: ...

    def judge(self, item: QuizItem, passage: str, image: bytes | None = None) -> JudgeVerdict:
        """``image`` (PNG bytes) is given for image questions so the judge can look at the
        figure the question is about."""
        ...


class ImageQuestionGenerator(Protocol):
    """Generates one question that requires interpreting a clinical figure."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        figure: FigureRef,
        image: bytes,
        section: Section,
        examples: list[SsmExample],
        label: SectionLabel | None,
    ) -> GeneratedBatch: ...


class StyleJudge(Protocol):
    """Eval-only judge scoring a question against the golden sets on a rubric."""

    @property
    def model_name(self) -> str: ...

    def score(
        self,
        item: QuizItem,
        golden_style: list[SsmExample],
        golden_regression: list[QuizItem],
        passage: str,
    ) -> StyleScores: ...


class ExampleSelector(Protocol):
    """Chooses which few-shot examples to show the generator for a given section."""

    @property
    def strategy(self) -> str: ...

    def select(self, query: str, k: int) -> list[SsmExample]: ...


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class UsageTracker(Protocol):
    """Receives one record per LLM/embedding call. Adapters call ``record``."""

    @property
    def run_id(self) -> str: ...

    @property
    def stage(self) -> str: ...

    def set_stage(self, stage: str) -> None: ...

    def set_context(self, *, unit_id: str = "", item_id: str = "") -> None: ...

    def record(self, record: UsageRecord) -> None: ...

    def records(self) -> list[UsageRecord]: ...


class PricingTable(Protocol):
    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float: ...


class ArtifactStore(Protocol):
    """Persists run artefacts (one JSON per stage) under a run id."""

    def new_run_id(self) -> str: ...

    def run_dir(self, run_id: str) -> Path: ...

    def list_runs(self) -> list[str]: ...

    def latest_run_id(self) -> str | None: ...

    def set_latest(self, run_id: str) -> None: ...

    def exists(self, run_id: str, name: str) -> bool: ...

    def save(self, run_id: str, name: str, model: BaseModel) -> Path: ...

    def load(self, run_id: str, name: str, model_type: type[M]) -> M: ...

    def save_list(self, run_id: str, name: str, models: list[BaseModel]) -> Path: ...

    def load_list(self, run_id: str, name: str, model_type: type[M]) -> list[M]: ...

    def append_jsonl(self, run_id: str, name: str, model: BaseModel) -> None: ...

    def read_jsonl(self, run_id: str, name: str, model_type: type[M]) -> list[M]: ...


class GoldenStore(Protocol):
    """Regression golden set: human-approved items promoted from the review UI."""

    def add(self, run_id: str, item: QuizItem) -> Path: ...

    def load_all(self) -> list[QuizItem]: ...


class QuizExporter(Protocol):
    @property
    def format_name(self) -> str: ...

    def export(self, items: list[QuizItem], target: Path) -> Path: ...
