"""A Container whose provider-backed adapters are replaced by fakes (no network)."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from medquiz_generator.application.context import RunContext
from medquiz_generator.composition.container import Container
from medquiz_generator.config.settings import Settings
from medquiz_generator.domain.models import StructuredDocument
from medquiz_generator.domain.ports import (
    DocumentParser,
    EmbeddingProvider,
    FigureTriage,
    ImageQuestionGenerator,
    QuestionGenerator,
    QuestionJudge,
    StyleJudge,
    TaxonomyBuilder,
    UsageTracker,
)
from medquiz_generator.domain.usage import ModelPrice, StaticPricingTable
from tests.fakes.fake_llm import (
    FakeEmbeddings,
    FakeFigureTriage,
    FakeGenerator,
    FakeImageQuestionGenerator,
    FakeJudge,
    FakeStyleJudge,
    FakeTaxonomyBuilder,
)
from tests.fakes.fake_parser import FakeParser

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "runs_dir": str(tmp_path / "runs"),
        "golden_dir": str(tmp_path / "golden"),
        "pricing_path": str(REPO_ROOT / "pricing.json"),
        "docling_cache_dir": str(tmp_path / "cache"),
        "langsmith_tracing": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


class FakeContainer(Container):
    def __init__(self, settings: Settings, document: StructuredDocument) -> None:
        super().__init__(settings)
        self.document = document
        self.fake_parser = FakeParser(document)

    @cached_property
    def pricing(self) -> StaticPricingTable:
        return StaticPricingTable(
            {
                "fake-generator": ModelPrice(input=1.0, output=2.0),
                "fake-judge": ModelPrice(input=1.0, output=2.0),
                "fake-embeddings": ModelPrice(input=0.5),
            }
        )

    def parser(self) -> DocumentParser:
        return self.fake_parser

    def generator(self, tracker: UsageTracker) -> QuestionGenerator:
        return FakeGenerator(tracker)

    def judge(self, tracker: UsageTracker) -> QuestionJudge | None:
        if not self.settings.verify_judge_enabled:
            return None
        return FakeJudge(tracker=tracker)

    def taxonomy_builder(self, tracker: UsageTracker) -> TaxonomyBuilder:
        return FakeTaxonomyBuilder()

    def style_judge(self, tracker: UsageTracker) -> StyleJudge:
        return FakeStyleJudge()

    def figure_triage(self, tracker: UsageTracker, ctx: RunContext | None = None) -> FigureTriage:
        return FakeFigureTriage(tracker)

    def image_generator(self, tracker: UsageTracker) -> ImageQuestionGenerator | None:
        if not self.settings.images_enabled:
            return None
        return FakeImageQuestionGenerator(tracker)

    def embedding_provider(self, tracker: UsageTracker) -> EmbeddingProvider:
        return FakeEmbeddings(tracker=tracker)

    def embeddings(self, tracker: UsageTracker) -> EmbeddingProvider | None:
        if not self.settings.dedup_enabled:
            return None
        return self.embedding_provider(tracker)
