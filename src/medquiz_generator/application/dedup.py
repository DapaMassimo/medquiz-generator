"""Stage 6 — dedup: embedding similarity between delivered questions; near-duplicates flagged.

Only items that survived verification are compared. When two questions exceed the threshold
the *later* one (document order) is flagged with ``duplicate_of`` pointing at the earlier.
"""

from __future__ import annotations

import math

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import (
    DedupReport,
    DuplicatePair,
    QuestionStatus,
    QuizItem,
    Stage,
)
from medquiz_generator.domain.ports import EmbeddingProvider


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def find_duplicates(
    ids: list[str], vectors: list[list[float]], threshold: float
) -> list[DuplicatePair]:
    """Pairwise scan; each item is reported at most once, against its earliest match."""
    pairs: list[DuplicatePair] = []
    flagged: set[str] = set()
    for j in range(len(ids)):
        if ids[j] in flagged:
            continue
        for i in range(j):
            if ids[i] in flagged:
                continue
            similarity = cosine(vectors[i], vectors[j])
            if similarity >= threshold:
                pairs.append(
                    DuplicatePair(item_id=ids[j], duplicate_of=ids[i], similarity=similarity)
                )
                flagged.add(ids[j])
                break
    return pairs


def dedup_text(item: QuizItem) -> str:
    correct = item.correct_option
    return f"{item.question}\nRisposta: {correct.text if correct else ''}"


class DedupStage:
    def __init__(self, embeddings: EmbeddingProvider | None, threshold: float = 0.92) -> None:
        self._embeddings = embeddings
        self._threshold = threshold

    def run(self, ctx: RunContext) -> list[QuizItem]:
        with ctx.stage(Stage.DEDUP):
            items = ctx.require_list(artifacts.QUESTIONS_VERIFIED, QuizItem)
            pairs: list[DuplicatePair] = []
            if self._embeddings is not None:
                candidates = [
                    i
                    for i in items
                    if i.status.status in (QuestionStatus.VERIFIED, QuestionStatus.FLAGGED)
                ]
                if candidates:
                    vectors = self._embeddings.embed([dedup_text(i) for i in candidates])
                    pairs = find_duplicates([i.id for i in candidates], vectors, self._threshold)
            by_id = {p.item_id: p for p in pairs}
            deduped = [
                item.with_status(
                    QuestionStatus.FLAGGED,
                    f"near-duplicate of {by_id[item.id].duplicate_of} "
                    f"(similarity {by_id[item.id].similarity:.2f})",
                    duplicate_of=by_id[item.id].duplicate_of,
                )
                if item.id in by_id
                else item
                for item in items
            ]
            ctx.store.save_list(ctx.run_id, artifacts.QUESTIONS_DEDUPED, list(deduped))
            ctx.store.save(
                ctx.run_id,
                artifacts.DEDUP_REPORT,
                DedupReport(run_id=ctx.run_id, threshold=self._threshold, pairs=pairs),
            )
            ctx.log("dedup finished", duplicates=len(pairs), enabled=self._embeddings is not None)
        return deduped
