"""Dependency injection: the only module that imports concrete adapters.

Everything is built from ``Settings``; use cases receive ports only. Adapters that need an
API key are created lazily, per run, so that CLI commands like ``plan`` or ``report`` never
touch a provider.
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

from medquiz_generator.adapters.embeddings.langchain_embeddings import (
    LangChainEmbeddings,
    build_embeddings,
)
from medquiz_generator.adapters.export.json_exporter import JsonExporter
from medquiz_generator.adapters.export.xlsx_exporter import XlsxExporter
from medquiz_generator.adapters.llm.adapters import (
    LangChainQuestionGenerator,
    LangChainQuestionJudge,
    LangChainStyleJudge,
    LangChainTaxonomyBuilder,
)
from medquiz_generator.adapters.llm.langchain_llm import StructuredLLM, build_chat_model
from medquiz_generator.adapters.parsing.docling_parser import DoclingParser
from medquiz_generator.adapters.storage.file_run_store import FileGoldenStore, FileRunStore
from medquiz_generator.adapters.tracing.langsmith import configure_tracing
from medquiz_generator.application import artifacts
from medquiz_generator.application.classify import ClassifyStage
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.dedup import DedupStage
from medquiz_generator.application.eval import EvalStage
from medquiz_generator.application.example_selection import (
    FirstNExampleSelector,
    SemanticExampleSelector,
)
from medquiz_generator.application.examples import load_ssm_examples
from medquiz_generator.application.export import ExportStage
from medquiz_generator.application.generate import GenerateStage
from medquiz_generator.application.parse import ParseStage
from medquiz_generator.application.pipeline import Pipeline, Stages
from medquiz_generator.application.plan import PlanStage
from medquiz_generator.application.report import ReportStage
from medquiz_generator.application.review import ReviewService
from medquiz_generator.application.verify import VerifyStage
from medquiz_generator.config.pricing import load_pricing
from medquiz_generator.config.settings import Settings
from medquiz_generator.domain.models import SsmExample
from medquiz_generator.domain.ports import (
    DocumentParser,
    EmbeddingProvider,
    ExampleSelector,
    QuestionGenerator,
    QuestionJudge,
    StyleJudge,
    TaxonomyBuilder,
    UsageTracker,
)
from medquiz_generator.domain.usage import InMemoryUsageTracker, StaticPricingTable
from medquiz_generator.resources import bundled_examples_path

log = logging.getLogger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.tracing_enabled = configure_tracing(
            enabled=self.settings.langsmith_tracing,
            api_key=self.settings.langsmith_api_key,
            project=self.settings.langsmith_project,
            endpoint=self.settings.langsmith_endpoint,
        )

    # ------------------------------------------------------------------ shared

    @cached_property
    def pricing(self) -> StaticPricingTable:
        return load_pricing(Path(self.settings.pricing_path))

    @cached_property
    def store(self) -> FileRunStore:
        return FileRunStore(Path(self.settings.runs_dir))

    @cached_property
    def golden_store(self) -> FileGoldenStore:
        return FileGoldenStore(Path(self.settings.golden_dir))

    @cached_property
    def examples(self) -> list[SsmExample]:
        configured = self.settings.examples_path
        return load_ssm_examples(Path(configured) if configured else bundled_examples_path())

    @cached_property
    def review(self) -> ReviewService:
        return ReviewService(self.store, self.golden_store)

    def new_run_id(self) -> str:
        run_id = self.store.new_run_id()
        self.store.run_dir(run_id)
        self.store.set_latest(run_id)
        return run_id

    def context(self, run_id: str) -> RunContext:
        tracker = InMemoryUsageTracker(
            run_id,
            self.pricing,
            sink=lambda record: self.store.append_jsonl(run_id, artifacts.USAGE_LOG, record),
        )
        return RunContext(run_id=run_id, store=self.store, tracker=tracker)

    # ---------------------------------------------------------------- adapters

    def parser(self) -> DocumentParser:
        s = self.settings
        return DoclingParser(
            cache_dir=Path(s.docling_cache_dir) if s.docling_cache_dir else None,
            artifacts_path=Path(s.docling_artifacts_path) if s.docling_artifacts_path else None,
            do_ocr=s.docling_do_ocr,
            ocr_engine=s.docling_ocr_engine,
        )

    def _api_key(self) -> str:
        s = self.settings
        return s.openai_api_key if s.llm_provider == "openai" else s.anthropic_api_key

    def structured_llm(self, model: str, tracker: UsageTracker | None) -> StructuredLLM:
        s = self.settings
        chat_model = build_chat_model(
            s.llm_provider,
            model,
            temperature=s.llm_temperature,
            api_key=self._api_key(),
            max_retries=s.llm_max_retries,
        )
        return StructuredLLM(chat_model, model_name=model, tracker=tracker)

    def generator(self, tracker: UsageTracker) -> QuestionGenerator:
        return LangChainQuestionGenerator(
            self.structured_llm(self.settings.generator_model, tracker)
        )

    def judge(self, tracker: UsageTracker) -> QuestionJudge | None:
        if not self.settings.verify_judge_enabled:
            return None
        return LangChainQuestionJudge(self.structured_llm(self.settings.judge_model, tracker))

    def taxonomy_builder(self, tracker: UsageTracker) -> TaxonomyBuilder:
        return LangChainTaxonomyBuilder(self.structured_llm(self.settings.taxonomy_model, tracker))

    def style_judge(self, tracker: UsageTracker) -> StyleJudge:
        return LangChainStyleJudge(self.structured_llm(self.settings.eval_judge_model, tracker))

    def embedding_provider(self, tracker: UsageTracker) -> EmbeddingProvider:
        s = self.settings
        return LangChainEmbeddings(
            build_embeddings("openai", s.embedding_model, api_key=s.openai_api_key),
            model_name=s.embedding_model,
            tracker=tracker,
        )

    def embeddings(self, tracker: UsageTracker) -> EmbeddingProvider | None:
        """Embeddings for the dedup stage (None when dedup is disabled)."""
        if not self.settings.dedup_enabled:
            return None
        return self.embedding_provider(tracker)

    def example_selector(self, tracker: UsageTracker) -> ExampleSelector:
        s = self.settings
        if s.examples_selection == "semantic" and len(self.examples) > s.max_examples_in_prompt:
            return SemanticExampleSelector(
                self.embedding_provider(tracker),
                self.examples,
                min_similarity=s.examples_min_similarity,
            )
        return FirstNExampleSelector(self.examples)

    # ------------------------------------------------------------------ stages

    def parse_stage(self) -> ParseStage:
        return ParseStage(self.parser())

    def plan_stage(self) -> PlanStage:
        return PlanStage(min_unit_chars=self.settings.plan_min_unit_chars)

    def classify_stage(self, ctx: RunContext) -> ClassifyStage:
        return ClassifyStage(self.taxonomy_builder(ctx.tracker))

    def generate_stage(self, ctx: RunContext) -> GenerateStage:
        return GenerateStage(
            self.generator(ctx.tracker),
            self.examples,
            max_examples=self.settings.max_examples_in_prompt,
            selector=self.example_selector(ctx.tracker),
        )

    def verify_stage(self, ctx: RunContext) -> VerifyStage:
        return VerifyStage(self.judge(ctx.tracker))

    def dedup_stage(self, ctx: RunContext) -> DedupStage:
        return DedupStage(self.embeddings(ctx.tracker), threshold=self.settings.dedup_threshold)

    def export_stage(self) -> ExportStage:
        return ExportStage(
            [JsonExporter(), XlsxExporter()],
            include_flagged=self.settings.export_include_flagged,
        )

    def report_stage(self) -> ReportStage:
        return ReportStage()

    def eval_stage(self, ctx: RunContext, *, include_flagged: bool = False) -> EvalStage:
        return EvalStage(
            self.style_judge(ctx.tracker),
            self.examples,
            self.golden_store,
            include_flagged=include_flagged,
        )

    def pipeline(self, ctx: RunContext) -> Pipeline:
        return Pipeline(
            Stages(
                parse=self.parse_stage(),
                plan=self.plan_stage(),
                classify=self.classify_stage(ctx),
                generate=self.generate_stage(ctx),
                verify=self.verify_stage(ctx),
                dedup=self.dedup_stage(ctx),
                export=self.export_stage(),
                report=self.report_stage(),
            )
        )
