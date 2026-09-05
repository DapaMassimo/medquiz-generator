"""OpenAI embeddings (via langchain-openai) behind the EmbeddingProvider port, with usage
tracking. Token counts use tiktoken when available, otherwise a chars/4 estimate."""

from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings

from medquiz_generator.domain.errors import LLMError
from medquiz_generator.domain.models import UsageRecord
from medquiz_generator.domain.ports import UsageTracker
from medquiz_generator.domain.text import estimate_tokens


def build_embeddings(provider: str, model: str, *, api_key: str) -> Embeddings:
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=api_key or None)  # type: ignore[arg-type]
    raise LLMError(f"Unsupported embedding provider: {provider}")


def count_tokens(texts: list[str], model: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # unknown model / tiktoken missing → estimate
        return sum(estimate_tokens(t) for t in texts)
    return sum(len(encoding.encode(t)) for t in texts)


class LangChainEmbeddings:
    def __init__(
        self, embeddings: Embeddings, *, model_name: str, tracker: UsageTracker | None = None
    ) -> None:
        self._embeddings = embeddings
        self._model_name = model_name
        self._tracker = tracker

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        try:
            vectors = self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise LLMError(f"Embedding call failed: {exc}") from exc
        if self._tracker is not None:
            self._tracker.record(
                UsageRecord(
                    run_id="",
                    stage="",
                    model=self._model_name,
                    kind="embedding",
                    input_tokens=count_tokens(texts, self._model_name),
                    duration_s=time.perf_counter() - t0,
                )
            )
        return [list(map(float, v)) for v in vectors]
