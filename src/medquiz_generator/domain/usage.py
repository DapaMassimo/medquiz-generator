"""Cost/usage tracking primitives (pure Python, no framework).

``InMemoryUsageTracker`` is the single place where every LLM/embedding call is recorded and
priced. Adapters only call ``record``; the application layer sets the current stage/unit/item
context so records can be aggregated per stage, per section and per question.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from medquiz_generator.domain.models import UsageRecord


class ModelPrice(BaseModel):
    """USD per one million tokens."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0, default=0.0)


class StaticPricingTable:
    """Prices loaded from a config file; unknown models cost 0 and are reported."""

    def __init__(self, prices: dict[str, ModelPrice], per_tokens: int = 1_000_000) -> None:
        self._prices = dict(prices)
        self._per_tokens = per_tokens
        self.unknown_models: set[str] = set()

    def price_for(self, model: str) -> ModelPrice | None:
        if model in self._prices:
            return self._prices[model]
        # Allow dated snapshots ("gpt-4o-mini-2024-07-18") to inherit the base price.
        candidates = [name for name in self._prices if model.startswith(name)]
        if candidates:
            return self._prices[max(candidates, key=len)]
        return None

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.price_for(model)
        if price is None:
            self.unknown_models.add(model)
            return 0.0
        return (input_tokens * price.input + output_tokens * price.output) / self._per_tokens

    @property
    def models(self) -> list[str]:
        return sorted(self._prices)


class InMemoryUsageTracker:
    """Collects UsageRecords; optionally forwards each one to a sink (e.g. usage.jsonl)."""

    def __init__(
        self,
        run_id: str,
        pricing: StaticPricingTable | None = None,
        sink: Callable[[UsageRecord], None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._pricing = pricing
        self._sink = sink
        self._stage = ""
        self._unit_id = ""
        self._item_id = ""
        self._records: list[UsageRecord] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def stage(self) -> str:
        return self._stage

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self._unit_id = ""
        self._item_id = ""

    def set_context(self, *, unit_id: str = "", item_id: str = "") -> None:
        self._unit_id = unit_id
        self._item_id = item_id

    def record(self, record: UsageRecord) -> None:
        updates: dict[str, object] = {}
        if not record.run_id:
            updates["run_id"] = self._run_id
        if not record.stage:
            updates["stage"] = self._stage
        if not record.unit_id:
            updates["unit_id"] = self._unit_id
        if not record.item_id:
            updates["item_id"] = self._item_id
        if record.cost_usd == 0.0 and self._pricing is not None:
            updates["cost_usd"] = self._pricing.cost(
                record.model, record.input_tokens, record.output_tokens
            )
        final = record.model_copy(update=updates) if updates else record
        self._records.append(final)
        if self._sink is not None:
            self._sink(final)

    def records(self) -> list[UsageRecord]:
        return list(self._records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)
