from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from medquiz_generator.adapters.tracing.langsmith import configure_tracing
from medquiz_generator.composition.container import Container
from medquiz_generator.config.pricing import load_pricing
from medquiz_generator.config.settings import Settings
from medquiz_generator.domain.models import (
    EvalComparison,
    EvalReport,
    GenerationPlan,
    PlanUnit,
    RunReport,
    Taxonomy,
    TaxonomySubject,
    TaxonomyTopic,
    UsageSummary,
)
from medquiz_generator.entrypoints.logsetup import (
    ContextFormatter,
    JsonFormatter,
    configure_logging,
)
from medquiz_generator.entrypoints.presentation import (
    render_comparison,
    render_eval,
    render_plan,
    render_report,
    render_taxonomy,
)
from tests.fakes.fake_container import make_settings


def test_settings_prefix_and_csv(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MQ_GENERATOR_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.generator_model == "gpt-4.1-mini"
    assert settings.openai_api_key == "sk-test"
    assert settings.docling_do_ocr is False
    assert settings.judge_model == "gpt-4o-mini"


def test_pricing_loader(tmp_path: Path) -> None:
    table = load_pricing(Path("pricing.json"))
    assert table.cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert load_pricing(tmp_path / "missing.json").cost("x", 10, 10) == 0.0


def test_container_wiring_from_settings(tmp_path: Path) -> None:
    container = Container(make_settings(tmp_path, verify_judge_enabled=False, dedup_enabled=False))
    run_id = container.new_run_id()
    ctx = container.context(run_id)
    assert container.judge(ctx.tracker) is None
    assert container.embeddings(ctx.tracker) is None
    assert container.store.latest_run_id() == run_id
    assert len(container.examples) == 19
    assert container.parser().name == "docling(ocr=off)"
    assert container.export_stage() is not None
    assert container.tracing_enabled is False


def test_container_builds_langchain_adapters_without_network(tmp_path: Path) -> None:
    container = Container(make_settings(tmp_path, openai_api_key="sk-test"))
    ctx = container.context(container.new_run_id())
    assert container.generator(ctx.tracker).model_name == "gpt-4o-mini"
    assert container.judge(ctx.tracker) is not None
    assert container.embeddings(ctx.tracker) is not None
    assert container.taxonomy_builder(ctx.tracker).model_name == "gpt-4o-mini"
    assert container.style_judge(ctx.tracker).model_name == "gpt-4o-mini"
    anthropic = Container(
        make_settings(
            tmp_path,
            llm_provider="anthropic",
            anthropic_api_key="sk-ant",
            generator_model="claude-sonnet-5",
        )
    )
    assert anthropic.generator(ctx.tracker).model_name == "claude-sonnet-5"


def test_configure_tracing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert configure_tracing(enabled=False, api_key="", project="p") is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert configure_tracing(enabled=True, api_key="", project="p") is False
    assert configure_tracing(enabled=True, api_key="ls-key", project="proj") is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_configure_tracing_exports_the_workspace_region(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A .env line only reaches LangChain if configure_tracing re-exports it."""
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
    eu = "https://eu.api.smith.langchain.com"
    assert configure_tracing(enabled=True, api_key="ls-key", project="p", endpoint=eu) is True
    assert os.environ["LANGSMITH_ENDPOINT"] == eu
    assert os.environ["LANGCHAIN_ENDPOINT"] == eu
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_configure_tracing_keeps_an_endpoint_already_in_the_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An empty setting must not wipe a region exported by the real environment."""
    exported = "https://eu.api.smith.langchain.com"
    monkeypatch.setenv("LANGSMITH_ENDPOINT", exported)
    assert configure_tracing(enabled=True, api_key="ls-key", project="p") is True
    assert os.environ["LANGSMITH_ENDPOINT"] == exported
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_settings_read_the_endpoint_under_its_standard_name(tmp_path: Path) -> None:
    """LANGSMITH_ENDPOINT keeps the vendor's name: no MQ_ prefix, like the API keys."""
    env = tmp_path / "regione.env"
    env.write_text("LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com\n", encoding="utf-8")
    settings = Settings(_env_file=str(env))  # type: ignore[call-arg]
    assert settings.langsmith_endpoint == "https://eu.api.smith.langchain.com"


def test_log_formatters() -> None:
    record = logging.LogRecord("medquiz", logging.INFO, __file__, 1, "hello", None, None)
    record.ctx = {"run_id": "r1", "stage": "plan", "empty": ""}  # type: ignore[attr-defined]
    text = ContextFormatter("%(message)s").format(record)
    assert text == "hello | run_id=r1 stage=plan"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello" and payload["run_id"] == "r1"
    configure_logging("DEBUG", json_logs=True)
    assert logging.getLogger().level == logging.DEBUG
    configure_logging("INFO")


def test_presentation_renderers() -> None:
    plan = GenerationPlan(
        run_id="r",
        document_hash="h",
        total_questions=3,
        min_unit_chars=600,
        units=[
            PlanUnit(
                id="u001",
                title="T",
                section_ids=["s1"],
                page_start=1,
                page_end=2,
                char_count=100,
                weight=1.0,
                questions=3,
            )
        ],
    )
    assert "3/3 questions" in render_plan(plan)
    taxonomy = Taxonomy(
        run_id="r",
        document_hash="h",
        subjects=[TaxonomySubject(name="S", topics=[TaxonomyTopic(name="T", section_ids=["s1"])])],
    )
    assert "- S" in render_taxonomy(taxonomy)
    report = RunReport(
        run_id="r",
        verified=2,
        flagged=1,
        discarded=1,
        discard_reasons={"judge: x": 1},
        flag_reasons={"dup": 1},
        deterministic_failures={"quote_missing": 1},
        judge_verdicts={"pass": 3},
        usage_by_stage={
            "generate": UsageSummary(calls=1, input_tokens=10, output_tokens=5, cost_usd=0.01)
        },
        usage_total=UsageSummary(calls=1, input_tokens=10, output_tokens=5, cost_usd=0.01),
    )
    text = render_report(report)
    assert "Cost per delivered question" in text and "quote_missing=1" in text
    eval_report = EvalReport(
        run_id="r",
        judge_model="m",
        golden_style_count=5,
        golden_regression_count=0,
        items=[],
        averages={"mean": 4.0},
    )
    assert "Eval for run r" in render_eval(eval_report)
    comparison = EvalComparison(
        run_a="a",
        run_b="b",
        averages_a={"mean": 4.0},
        averages_b={"mean": 4.5},
        delta={"mean": 0.5},
    )
    assert "+0.50" in render_comparison(comparison)
