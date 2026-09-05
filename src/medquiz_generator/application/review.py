"""Human review: approve/reject decisions persisted per run, promotion to the golden set."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from medquiz_generator.application import artifacts
from medquiz_generator.domain.errors import MedQuizError
from medquiz_generator.domain.models import QuestionStatus, QuizItem, ReviewDecision, ReviewState
from medquiz_generator.domain.ports import ArtifactStore, GoldenStore


class ReviewService:
    def __init__(self, store: ArtifactStore, golden: GoldenStore) -> None:
        self._store = store
        self._golden = golden

    def state(self, run_id: str) -> ReviewState:
        if self._store.exists(run_id, artifacts.REVIEW_STATE):
            return self._store.load(run_id, artifacts.REVIEW_STATE, ReviewState)
        return ReviewState(run_id=run_id)

    def items(self, run_id: str) -> list[QuizItem]:
        for name in (
            artifacts.QUESTIONS_DEDUPED,
            artifacts.QUESTIONS_VERIFIED,
            artifacts.QUESTIONS_GENERATED,
        ):
            if self._store.exists(run_id, name):
                return self._store.load_list(run_id, name, QuizItem)
        return []

    def decide(
        self,
        run_id: str,
        item_id: str,
        decision: Literal["approved", "rejected"],
        note: str = "",
    ) -> ReviewState:
        state = self.state(run_id)
        state.decisions[item_id] = ReviewDecision(item_id=item_id, decision=decision, note=note)
        self._store.save(run_id, artifacts.REVIEW_STATE, state)
        return state

    def clear(self, run_id: str, item_id: str) -> ReviewState:
        state = self.state(run_id)
        state.decisions.pop(item_id, None)
        self._store.save(run_id, artifacts.REVIEW_STATE, state)
        return state

    def approved_items(self, run_id: str) -> list[QuizItem]:
        state = self.state(run_id)
        return [
            i
            for i in self.items(run_id)
            if (d := state.decisions.get(i.id)) is not None and d.decision == "approved"
        ]

    def promote(self, run_id: str, item_id: str) -> Path:
        """Promote a verified-and-approved item to the regression golden set."""
        item = next((i for i in self.items(run_id) if i.id == item_id), None)
        if item is None:
            raise MedQuizError(f"Item {item_id} not found in run {run_id}.")
        if item.status.status not in (QuestionStatus.VERIFIED, QuestionStatus.FLAGGED):
            raise MedQuizError(f"Item {item_id} did not pass verification; cannot promote.")
        decision = self.state(run_id).decisions.get(item_id)
        if decision is None or decision.decision != "approved":
            raise MedQuizError(f"Item {item_id} must be approved before promotion.")
        return self._golden.add(run_id, item)
