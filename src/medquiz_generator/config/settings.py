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
    examples_path: str = "examples/ssm/ssm_examples.txt"
    pricing_path: str = "pricing.json"

    # Parsing (Docling)
    docling_cache_dir: str | None = ".cache/docling"
    docling_artifacts_path: str | None = None
    docling_do_ocr: bool = False
    # Allowlisted engines only (rapidocr / easyocr / tesseract): see docling_parser.py.
    docling_ocr_engine: str = "rapidocr"
    use_cache: bool = True

    # Pipeline knobs
    plan_min_unit_chars: int = 600
    max_examples_in_prompt: int = 5
    verify_judge_enabled: bool = True
    dedup_enabled: bool = True
    dedup_threshold: float = 0.92
    export_include_flagged: bool = False

    # Observability
    langsmith_tracing: bool = False
    langsmith_project: str = "medquiz-generator"
    log_level: str = "INFO"
    log_json: bool = False
