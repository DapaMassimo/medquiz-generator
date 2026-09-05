"""File-based artefact store: ``runs/<run_id>/<name>.json`` (+ ``.jsonl`` logs).

No database: every artefact is a JSON file carrying the run id, so the full lineage
input → section → question can be inspected with any text editor.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from medquiz_generator.domain.models import GoldenItem, QuizItem

M = TypeVar("M", bound=BaseModel)

_LATEST = "LATEST"


class FileRunStore:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- run ids

    def new_run_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    def run_dir(self, run_id: str) -> Path:
        path = self._base / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self._base.iterdir() if p.is_dir())

    def latest_run_id(self) -> str | None:
        pointer = self._base / _LATEST
        if pointer.exists():
            run_id = pointer.read_text(encoding="utf-8").strip()
            if run_id and (self._base / run_id).is_dir():
                return run_id
        runs = self.list_runs()
        return runs[-1] if runs else None

    def set_latest(self, run_id: str) -> None:
        (self._base / _LATEST).write_text(run_id, encoding="utf-8")

    # ---------------------------------------------------------------- artefacts

    def _path(self, run_id: str, name: str, suffix: str = ".json") -> Path:
        return self.run_dir(run_id) / f"{name}{suffix}"

    def exists(self, run_id: str, name: str) -> bool:
        return self._path(run_id, name).exists() or self._path(run_id, name, ".jsonl").exists()

    def save(self, run_id: str, name: str, model: BaseModel) -> Path:
        path = self._path(run_id, name)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str, name: str, model_type: type[M]) -> M:
        path = self._path(run_id, name)
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def save_list(self, run_id: str, name: str, models: list[BaseModel]) -> Path:
        path = self._path(run_id, name)
        payload = [m.model_dump(mode="json") for m in models]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_list(self, run_id: str, name: str, model_type: type[M]) -> list[M]:
        path = self._path(run_id, name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [model_type.model_validate(entry) for entry in payload]

    def append_jsonl(self, run_id: str, name: str, model: BaseModel) -> None:
        path = self._path(run_id, name, ".jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(model.model_dump_json() + "\n")

    def read_jsonl(self, run_id: str, name: str, model_type: type[M]) -> list[M]:
        path = self._path(run_id, name, ".jsonl")
        if not path.exists():
            return []
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [model_type.model_validate_json(line) for line in lines]


class FileGoldenStore:
    """``golden/regression/<run_id>__<item_id>.json`` — one promoted item per file."""

    def __init__(self, golden_dir: Path) -> None:
        self._dir = golden_dir / "regression"
        self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, run_id: str, item: QuizItem) -> Path:
        path = self._dir / f"{run_id}__{item.id}.json"
        path.write_text(
            GoldenItem(run_id=run_id, item=item).model_dump_json(indent=2), encoding="utf-8"
        )
        return path

    def load_all(self) -> list[QuizItem]:
        items: list[QuizItem] = []
        for path in sorted(self._dir.glob("*.json")):
            golden = GoldenItem.model_validate_json(path.read_text(encoding="utf-8"))
            items.append(golden.item)
        return items
