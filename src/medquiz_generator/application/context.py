"""Per-run context shared by all stages: run id, artefact store, usage tracker, logger."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal, TypeVar

from pydantic import BaseModel

from medquiz_generator.application import artifacts
from medquiz_generator.domain.errors import StageInputMissingError
from medquiz_generator.domain.models import RunMeta, StageTiming
from medquiz_generator.domain.ports import ArtifactStore, UsageTracker

M = TypeVar("M", bound=BaseModel)


class RunContext:
    def __init__(
        self,
        run_id: str,
        store: ArtifactStore,
        tracker: UsageTracker,
        logger: logging.Logger | None = None,
    ) -> None:
        self.run_id = run_id
        self.store = store
        self.tracker = tracker
        self.logger = logger or logging.getLogger("medquiz")

    # ------------------------------------------------------------------ logging

    def log(self, message: str, **fields: object) -> None:
        extra = {"run_id": self.run_id, "stage": self.tracker.stage, **fields}
        self.logger.info(message, extra={"ctx": extra})

    def warn(self, message: str, **fields: object) -> None:
        extra = {"run_id": self.run_id, "stage": self.tracker.stage, **fields}
        self.logger.warning(message, extra={"ctx": extra})

    # ------------------------------------------------------------------ artefacts

    def require(self, name: str, model_type: type[M]) -> M:
        if not self.store.exists(self.run_id, name):
            raise StageInputMissingError(
                f"Run {self.run_id}: artefact '{name}' not found. Run the previous stage first."
            )
        return self.store.load(self.run_id, name, model_type)

    def require_list(self, name: str, model_type: type[M]) -> list[M]:
        if not self.store.exists(self.run_id, name):
            raise StageInputMissingError(
                f"Run {self.run_id}: artefact '{name}' not found. Run the previous stage first."
            )
        return self.store.load_list(self.run_id, name, model_type)

    def meta(self) -> RunMeta:
        if self.store.exists(self.run_id, artifacts.RUN_META):
            return self.store.load(self.run_id, artifacts.RUN_META, RunMeta)
        return RunMeta(run_id=self.run_id)

    def update_meta(self, **fields: object) -> RunMeta:
        meta = self.meta().model_copy(update=fields)
        self.store.save(self.run_id, artifacts.RUN_META, meta)
        return meta

    # ------------------------------------------------------------------ stages

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        """Set the tracker stage, time the block and append a StageTiming record."""
        self.tracker.set_stage(name)
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        self.log(f"stage {name} started")
        status: Literal["ok", "failed"] = "ok"
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"[:2000]
            self.logger.error(
                f"stage {name} failed: {error}", extra={"ctx": {"run_id": self.run_id}}
            )
            raise
        finally:
            duration = time.perf_counter() - t0
            timing = StageTiming(
                stage=name,
                duration_s=duration,
                started_at=started,
                finished_at=datetime.now(UTC),
                status=status,
                error=error,
            )
            self.store.append_jsonl(self.run_id, artifacts.TIMINGS_LOG, timing)
            self.log(f"stage {name} finished", duration_s=round(duration, 3))
            self.tracker.set_stage("")
