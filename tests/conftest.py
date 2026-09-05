from __future__ import annotations

import json
from pathlib import Path

import pytest

from medquiz_generator.adapters.storage.file_run_store import FileGoldenStore, FileRunStore
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import StructuredDocument
from medquiz_generator.domain.usage import InMemoryUsageTracker, ModelPrice, StaticPricingTable

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES_PATH = Path(__file__).parent.parent / "examples" / "ssm" / "ssm_examples.txt"


@pytest.fixture(scope="session")
def fixture_document() -> StructuredDocument:
    payload = json.loads((FIXTURES / "structured_document.json").read_text(encoding="utf-8"))
    return StructuredDocument.model_validate(payload)


@pytest.fixture
def store(tmp_path: Path) -> FileRunStore:
    return FileRunStore(tmp_path / "runs")


@pytest.fixture
def golden_store(tmp_path: Path) -> FileGoldenStore:
    return FileGoldenStore(tmp_path / "golden")


@pytest.fixture
def pricing() -> StaticPricingTable:
    return StaticPricingTable(
        {
            "fake-generator": ModelPrice(input=1.0, output=2.0),
            "fake-judge": ModelPrice(input=1.0, output=2.0),
            "fake-embeddings": ModelPrice(input=0.5),
        }
    )


@pytest.fixture
def ctx(store: FileRunStore, pricing: StaticPricingTable) -> RunContext:
    run_id = "run-test"
    tracker = InMemoryUsageTracker(
        run_id, pricing, sink=lambda rec: store.append_jsonl(run_id, "usage", rec)
    )
    return RunContext(run_id=run_id, store=store, tracker=tracker)
