"""LangChain-backed structured LLM calls with usage tracking and retries.

``build_chat_model`` is the single place where provider/model selection is materialised;
``StructuredLLM.invoke`` wraps ``with_structured_output`` so every adapter gets the same
retry policy, token accounting and cost attribution for free.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from medquiz_generator.domain.errors import LLMError
from medquiz_generator.domain.models import UsageRecord
from medquiz_generator.domain.ports import UsageTracker

M = TypeVar("M", bound=BaseModel)
log = logging.getLogger(__name__)


# OpenAI reasoning models (o-series, gpt-5 family) reject any temperature other than the
# default; the sampling temperature is therefore only sent to non-reasoning models.
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-5)")


def is_reasoning_model(model: str) -> bool:
    return bool(_OPENAI_REASONING.match(model))


def build_chat_model(
    provider: str, model: str, *, temperature: float, api_key: str, max_retries: int = 2
) -> BaseChatModel:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {}
        if not is_reasoning_model(model):
            kwargs["temperature"] = temperature
        return ChatOpenAI(
            model=model,
            api_key=api_key or None,  # type: ignore[arg-type]
            max_retries=max_retries,
            **kwargs,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            api_key=api_key or None,  # type: ignore[arg-type]
            max_retries=max_retries,
            timeout=None,
            stop=None,
        )
    raise LLMError(f"Unsupported LLM provider: {provider}")


def _usage_from(raw: Any) -> tuple[int, int, str | None]:
    usage = getattr(raw, "usage_metadata", None) or {}
    metadata = getattr(raw, "response_metadata", None) or {}
    model_name = metadata.get("model_name") or metadata.get("model")
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), model_name


class StructuredLLM:
    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        model_name: str,
        tracker: UsageTracker | None = None,
        max_attempts: int = 3,
        backoff_s: float = 1.5,
    ) -> None:
        self._model = chat_model
        self._model_name = model_name
        self._tracker = tracker
        self._max_attempts = max(1, max_attempts)
        self._backoff_s = backoff_s

    @property
    def model_name(self) -> str:
        return self._model_name

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[M],
        *,
        images: list[bytes] | None = None,
    ) -> M:
        """``images`` (PNG bytes) are attached to the user message for vision-capable models."""
        runnable = self._model.with_structured_output(schema, include_raw=True)
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for png in images:
                encoded = base64.b64encode(png).decode("ascii")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                )
            human = HumanMessage(content=content)  # type: ignore[arg-type]
        else:
            human = HumanMessage(content=user_prompt)
        messages = [SystemMessage(content=system_prompt), human]
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            t0 = time.perf_counter()
            try:
                result = runnable.invoke(messages)
            except Exception as exc:  # provider/network errors: retry with backoff
                last_error = exc
                log.warning("LLM call failed (attempt %d/%d): %s", attempt, self._max_attempts, exc)
                time.sleep(self._backoff_s * attempt)
                continue
            duration = time.perf_counter() - t0
            raw = result.get("raw") if isinstance(result, dict) else None
            parsed = result.get("parsed") if isinstance(result, dict) else result
            self._record(raw, duration)
            if isinstance(parsed, schema):
                return parsed
            error = result.get("parsing_error") if isinstance(result, dict) else None
            last_error = LLMError(f"Structured output could not be parsed: {error}")
            log.warning("Structured output parse failure (attempt %d): %s", attempt, error)
        raise LLMError(f"LLM call failed after {self._max_attempts} attempts: {last_error}")

    def _record(self, raw: Any, duration: float) -> None:
        if self._tracker is None:
            return
        input_tokens, output_tokens, reported_model = _usage_from(raw)
        self._tracker.record(
            UsageRecord(
                run_id="",
                stage="",
                model=reported_model or self._model_name,
                kind="chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_s=duration,
            )
        )
