"""Loads the model price table (USD per 1M tokens) from a JSON file.

Editing ``pricing.json`` is enough to update costs — no code change required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from medquiz_generator.domain.usage import ModelPrice, StaticPricingTable

log = logging.getLogger(__name__)


class PricingFile(BaseModel):
    currency: str = "USD"
    per_tokens: int = Field(default=1_000_000, gt=0)
    models: dict[str, ModelPrice]


def load_pricing(path: Path) -> StaticPricingTable:
    if not path.exists():
        log.warning("Pricing table %s not found: all costs will be reported as 0.", path)
        return StaticPricingTable({})
    data = PricingFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return StaticPricingTable(data.models, per_tokens=data.per_tokens)
