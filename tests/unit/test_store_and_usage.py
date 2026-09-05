from __future__ import annotations

from pathlib import Path

from medquiz_generator.adapters.storage.file_run_store import FileGoldenStore, FileRunStore
from medquiz_generator.domain.models import (
    Difficulty,
    ModelMetadata,
    Option,
    QuizItem,
    RunMeta,
    SourceRef,
    UsageRecord,
)
from medquiz_generator.domain.usage import InMemoryUsageTracker, ModelPrice, StaticPricingTable


def make_item(item_id: str = "q1") -> QuizItem:
    return QuizItem(
        id=item_id,
        materia="Cardiologia",
        argomento="Ipertensione",
        question="Domanda?",
        options=[Option(id=o, text=f"opt {o}") for o in ("A", "B", "C", "D", "E")],
        correct_option_id="A",
        explanation="Perché sì.",
        difficulty=Difficulty.MEDIA,
        source=SourceRef(
            section_id="s001",
            section_title="T",
            section_path="Doc > T",
            page_start=1,
            page_end=2,
            page=1,
            quote="citazione",
        ),
        model_metadata=ModelMetadata(model="fake", run_id="r"),
    )


def test_run_store_roundtrip_and_latest(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs")
    run_id = store.new_run_id()
    assert not store.exists(run_id, "run_meta")
    path = store.save(run_id, "run_meta", RunMeta(run_id=run_id, total_questions=3))
    assert path.name == "run_meta.json"
    assert store.load(run_id, "run_meta", RunMeta).total_questions == 3
    store.save_list(run_id, "items", [make_item("a"), make_item("b")])
    assert [i.id for i in store.load_list(run_id, "items", QuizItem)] == ["a", "b"]
    store.append_jsonl(run_id, "usage", UsageRecord(run_id=run_id, stage="x", model="m"))
    store.append_jsonl(run_id, "usage", UsageRecord(run_id=run_id, stage="y", model="m"))
    assert [r.stage for r in store.read_jsonl(run_id, "usage", UsageRecord)] == ["x", "y"]
    assert store.read_jsonl(run_id, "nothing", UsageRecord) == []
    assert store.latest_run_id() == run_id
    store.set_latest(run_id)
    assert store.latest_run_id() == run_id
    assert store.list_runs() == [run_id]


def test_golden_store_roundtrip(tmp_path: Path) -> None:
    golden = FileGoldenStore(tmp_path / "golden")
    assert golden.load_all() == []
    golden.add("run-1", make_item("q7"))
    items = golden.load_all()
    assert [i.id for i in items] == ["q7"]


def test_pricing_table_costs_and_prefix_match() -> None:
    table = StaticPricingTable({"gpt-4o-mini": ModelPrice(input=0.15, output=0.6)})
    assert table.cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert table.cost("gpt-4o-mini-2024-07-18", 2_000_000, 0) == 0.30
    assert table.cost("unknown-model", 100, 100) == 0.0
    assert table.unknown_models == {"unknown-model"}
    assert table.models == ["gpt-4o-mini"]


def test_tracker_fills_context_and_prices_records() -> None:
    table = StaticPricingTable({"m": ModelPrice(input=1.0, output=2.0)})
    sink: list[UsageRecord] = []
    tracker = InMemoryUsageTracker("run-9", table, sink=sink.append)
    tracker.set_stage("generate")
    tracker.set_context(unit_id="u001", item_id="q1")
    tracker.record(
        UsageRecord(run_id="", stage="", model="m", input_tokens=1_000_000, output_tokens=500_000)
    )
    [record] = tracker.records()
    assert record.run_id == "run-9" and record.stage == "generate"
    assert record.unit_id == "u001" and record.item_id == "q1"
    assert record.cost_usd == 2.0
    assert sink == [record]
    tracker.set_stage("verify")
    assert tracker.stage == "verify"
    tracker.record(UsageRecord(run_id="", stage="", model="m", input_tokens=0, output_tokens=0))
    assert tracker.records()[1].unit_id == ""
    assert tracker.total_cost_usd == 2.0
