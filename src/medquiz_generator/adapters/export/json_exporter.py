"""JSON export: the full QuizItem schema, one array, UTF-8, human-readable."""

from __future__ import annotations

import json
from pathlib import Path

from medquiz_generator.domain.errors import ExportError
from medquiz_generator.domain.models import QuizItem


class JsonExporter:
    @property
    def format_name(self) -> str:
        return "json"

    def export(self, items: list[QuizItem], target: Path) -> Path:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = [item.model_dump(mode="json") for item in items]
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Cannot write {target}: {exc}") from exc
        return target
