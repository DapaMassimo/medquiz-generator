"""Application configuration via pydantic-settings (env / .env, prefix ``MQ_``)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MQ_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Provider keys (no prefix, standard names)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
    # Region of the LangSmith workspace, e.g. https://eu.api.smith.langchain.com for the EU.
    # Empty keeps LangSmith's own default (US).
    langsmith_endpoint: str = Field(default="", validation_alias="LANGSMITH_ENDPOINT")

    # Models — the single place where provider/model choices live.
    llm_provider: Literal["openai", "anthropic"] = "openai"
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    taxonomy_model: str = "gpt-4o-mini"
    eval_judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    llm_models: Annotated[list[str], NoDecode] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]

    @field_validator("llm_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # Paths
    runs_dir: str = "runs"
    golden_dir: str = "golden"
    # Empty means the file bundled in the wheel; set it to override with your own.
    examples_path: str = ""
    pricing_path: str = "pricing.json"

    # Parsing (Docling)
    docling_cache_dir: str | None = ".cache/docling"
    docling_artifacts_path: str | None = None
    docling_do_ocr: bool = False
    # Allowlisted engines only (rapidocr / easyocr / tesseract): see docling_parser.py.
    docling_ocr_engine: str = "rapidocr"
    use_cache: bool = True

    # Pipeline knobs
    # Merge threshold for small sibling sections: an integer, or "auto" to derive it from the
    # question budget so that every unit can get at least one question.
    plan_min_unit_chars: int | Literal["auto"] = "auto"
    # Few-shot examples per generation call: an upper bound, between 3 and 5.
    max_examples_in_prompt: int = Field(default=5, ge=3, le=5)
    # Share of question styles, in percent: "vignetta=70,nozionistica=30".
    style_mix: str = "vignetta=70,nozionistica=30"
    # Figures: extract them from the PDF as run assets and triage them with a vision model.
    images_enabled: bool = False
    images_min_side_px: int = 80
    vision_model: str = "gpt-4.1"
    # Image questions never reach "verified" on their own: a reviewer must confirm them.
    image_questions_require_review: bool = True
    # 'random' = k drawn from the whole pool, seeded by the section so re-runs reproduce;
    # 'first' = the first N of the file, the same for every unit; 'semantic' = closest to the
    # section by embedding, used only when the pool is larger than max_examples_in_prompt.
    # Semantic stays off until the pool is large enough to hold real neighbours: below that
    # most slots come from the fallback and it converges on 'first' while paying for the
    # embeddings anyway.
    examples_selection: Literal["first", "semantic", "random"] = "random"
    # Cosine similarity below which a semantic match is not convincing; missing slots are then
    # filled with the first examples of the file (FirstN fallback).
    examples_min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    verify_judge_enabled: bool = True
    dedup_enabled: bool = True
    dedup_threshold: float = 0.92
    export_include_flagged: bool = False

    # Observability
    langsmith_tracing: bool = False
    langsmith_project: str = "medquiz-generator"
    log_level: str = "INFO"
    log_json: bool = False
