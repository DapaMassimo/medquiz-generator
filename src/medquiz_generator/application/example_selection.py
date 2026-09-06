"""Few-shot example selection for the generator.

Three strategies, all returning a plain list of examples — selection never touches the prompts.

* ``random`` (default): k examples drawn from the whole pool, the draw seeded by the section
  so a re-run reproduces it. Every unit of the document sees a different slice of the pool.
* ``first``: the first k of the file, in file order. Every unit sees the same k.
* ``semantic``: the k nearest to the section by embedding similarity, keeping both registers
  — clinical vignette and direct factual question — represented, with slots that clear no
  similarity threshold filled from the head of the file.

Semantic is off by default. It only pays off once the pool is large enough to hold genuine
neighbours for most sections: below that, few candidates clear the threshold, most slots come
from the fallback, and the result converges on ``first`` while still paying for the
embeddings. Turn it on with ``MQ_EXAMPLES_SELECTION=semantic`` after the pool grows.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re

from medquiz_generator.application.dedup import cosine
from medquiz_generator.domain.models import SsmExample
from medquiz_generator.domain.ports import EmbeddingProvider

_VIGNETTE_START = re.compile(
    r"^\s*(un|una|il|la|l')\s*"
    r"(uomo|donna|paziente|bambin[oa]|ragazz[oa]|neonat[oa]|lattante|anzian[oa]|signor[ae]?|"
    r"adolescente|gestante|primipara|multipara)\b",
    re.IGNORECASE,
)
_AGE = re.compile(r"\bdi\s+\d{1,3}\s+(anni|mesi|giorni|settimane)\b", re.IGNORECASE)

log = logging.getLogger(__name__)


def is_clinical_vignette(question: str) -> bool:
    """Heuristic register detector: a vignette opens with a patient and/or states an age."""
    return bool(_VIGNETTE_START.match(question)) or bool(_AGE.search(question))


def example_text(example: SsmExample) -> str:
    correct = next((o.text for o in example.options if o.id == example.correct_option_id), "")
    return f"{example.question}\n{correct}"


class FirstNExampleSelector:
    """Default strategy: the first k examples of the pool, in file order."""

    def __init__(self, examples: list[SsmExample]) -> None:
        self._examples = list(examples)

    @property
    def strategy(self) -> str:
        return "first"

    def select(self, query: str, k: int) -> list[SsmExample]:
        return self._examples[: max(0, k)]


class RandomExampleSelector:
    """k examples drawn from the whole pool, deterministically per section.

    The draw is seeded from the section text, which buys both properties at once: a given
    section always gets the same examples, so re-running the generate stage reproduces the
    run, while different sections get different draws. FirstN gives the second property up
    entirely — with 19 examples and k=5 it shows the same five questions to every unit of
    the document, and the other fourteen never reach the model.

    The seed comes from blake2b rather than ``hash()``, which is salted per process and would
    make the same run irreproducible across invocations. The draw uses its own Random
    instance so it stays independent of any other use of the module-level generator.
    """

    def __init__(self, examples: list[SsmExample]) -> None:
        self._examples = list(examples)

    @property
    def strategy(self) -> str:
        return "random"

    def select(self, query: str, k: int) -> list[SsmExample]:
        if k <= 0 or not self._examples:
            return []
        if len(self._examples) <= k:
            return list(self._examples)
        digest = hashlib.blake2b(query.encode("utf-8"), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        return rng.sample(self._examples, k)


class SemanticExampleSelector:
    """Top-k examples by embedding similarity to the section, with register diversity."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        examples: list[SsmExample],
        *,
        min_similarity: float = 0.3,
        query_chars: int = 1500,
        ensure_both_registers: bool = True,
    ) -> None:
        self._embeddings = embeddings
        self._examples = list(examples)
        self._min_similarity = min_similarity
        self._query_chars = query_chars
        self._ensure_both = ensure_both_registers
        self._pool_vectors: list[list[float]] | None = None
        self.last_fallback_count = 0

    @property
    def strategy(self) -> str:
        return "semantic"

    def _vectors(self) -> list[list[float]]:
        # The pool is embedded once per selector instance (i.e. once per run).
        if self._pool_vectors is None:
            self._pool_vectors = self._embeddings.embed([example_text(e) for e in self._examples])
        return self._pool_vectors

    def select(self, query: str, k: int) -> list[SsmExample]:
        pool = self._examples
        if k <= 0 or not pool:
            return []
        if len(pool) <= k:
            return list(pool)
        vectors = self._vectors()
        [query_vector] = self._embeddings.embed([query[: self._query_chars]])
        similarity = [cosine(query_vector, v) for v in vectors]
        ranked = sorted(range(len(pool)), key=lambda i: (-similarity[i], i))
        convincing = [i for i in ranked if similarity[i] >= self._min_similarity]
        chosen = convincing[:k]
        if self._ensure_both and len(chosen) == k:
            registers_in_pool = {is_clinical_vignette(e.question) for e in pool}
            registers_chosen = {is_clinical_vignette(pool[i].question) for i in chosen}
            missing = registers_in_pool - registers_chosen
            if missing:
                wanted = missing.pop()
                best = next(
                    (i for i in convincing if is_clinical_vignette(pool[i].question) == wanted),
                    None,
                )
                if best is not None:
                    chosen[-1] = best
        # Fallback: not enough convincing neighbours → complete with the first examples of
        # the file, in file order, skipping the ones already chosen.
        self.last_fallback_count = 0
        if len(chosen) < k:
            for index in range(len(pool)):
                if len(chosen) >= k:
                    break
                if index not in chosen:
                    chosen.append(index)
                    self.last_fallback_count += 1
            log.info(
                "semantic example selection: %d convincing match(es) above %.2f, "
                "%d slot(s) filled from the first examples of the file",
                k - self.last_fallback_count,
                self._min_similarity,
                self.last_fallback_count,
            )
        return [pool[i] for i in chosen]
