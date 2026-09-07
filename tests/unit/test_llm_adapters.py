from __future__ import annotations

from typing import Any

import pytest

from medquiz_generator.adapters.embeddings.langchain_embeddings import (
    LangChainEmbeddings,
    count_tokens,
)
from medquiz_generator.adapters.llm import prompts
from medquiz_generator.adapters.llm.adapters import (
    LangChainQuestionGenerator,
    LangChainQuestionJudge,
    LangChainStyleJudge,
    LangChainTaxonomyBuilder,
)
from medquiz_generator.adapters.llm.langchain_llm import StructuredLLM, build_chat_model
from medquiz_generator.application.examples import parse_ssm_examples
from medquiz_generator.domain.errors import LLMError
from medquiz_generator.domain.models import PlanUnit, StructuredDocument, UsageRecord
from medquiz_generator.domain.usage import InMemoryUsageTracker
from tests.unit.test_verify import good_item


class FakeAIMessage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 20) -> None:
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.response_metadata = {"model_name": "fake-model-2026"}


class FakeRunnable:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.invocations: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.invocations.append(messages)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(self, responses: list[Any]) -> None:
        self.runnable = FakeRunnable(responses)
        self.schemas: list[type] = []

    def with_structured_output(self, schema: type, *, include_raw: bool = False) -> FakeRunnable:
        self.schemas.append(schema)
        return self.runnable


def structured(responses: list[Any], tracker: InMemoryUsageTracker | None = None) -> StructuredLLM:
    model = FakeChatModel(responses)
    return StructuredLLM(model, model_name="fake", tracker=tracker, max_attempts=2, backoff_s=0)  # type: ignore[arg-type]


def test_structured_llm_records_usage_and_returns_parsed() -> None:
    tracker = InMemoryUsageTracker("r")
    tracker.set_stage("verify")
    verdict = prompts.WireVerdict(
        answer_supported=True,
        single_correct=True,
        distractors_wrong=True,
        verdict="pass",
        reason="ok",
    )
    llm = structured([{"raw": FakeAIMessage(), "parsed": verdict, "parsing_error": None}], tracker)
    result = llm.invoke("sys", "user", prompts.WireVerdict)
    assert result.verdict == "pass"
    [record] = tracker.records()
    assert (record.input_tokens, record.output_tokens) == (100, 20)
    assert record.model == "fake-model-2026" and record.stage == "verify"


def test_structured_llm_retries_then_fails() -> None:
    llm = structured(
        [
            RuntimeError("boom"),
            {"raw": FakeAIMessage(), "parsed": None, "parsing_error": ValueError("bad")},
        ]
    )
    with pytest.raises(LLMError):
        llm.invoke("sys", "user", prompts.WireVerdict)


def test_structured_llm_retries_then_succeeds() -> None:
    verdict = prompts.WireVerdict(
        answer_supported=True,
        single_correct=True,
        distractors_wrong=True,
        verdict="flag",
        reason="x",
    )
    llm = structured(
        [RuntimeError("boom"), {"raw": FakeAIMessage(), "parsed": verdict, "parsing_error": None}]
    )
    assert llm.invoke("s", "u", prompts.WireVerdict).verdict == "flag"


def test_build_chat_model_rejects_unknown_provider() -> None:
    with pytest.raises(LLMError):
        build_chat_model("nope", "m", temperature=0.0, api_key="")


def test_generator_adapter_maps_wire_batch(fixture_document: StructuredDocument) -> None:
    section = fixture_document.sections[3]
    unit = PlanUnit(
        id="u001",
        title=section.title,
        section_ids=[section.id],
        page_start=2,
        page_end=3,
        char_count=len(section.text),
        weight=0.2,
        questions=1,
    )
    wire = prompts.WireBatch(
        questions=[
            prompts.WireQuestion(
                question="Domanda?",
                options=[
                    prompts.WireOption(id=i, text=f"opt {i} [corretta]" if i == "B" else f"opt {i}")  # type: ignore[arg-type]
                    for i in "ABCDE"
                ],
                correct_option_id="B",
                explanation="Spiegazione lunga a sufficienza per il controllo.",
                difficulty="difficile",
                quote="La diagnosi richiede misurazioni ripetute",
                page=2,
            )
        ]
    )
    llm = structured([{"raw": FakeAIMessage(), "parsed": wire, "parsing_error": None}])
    examples = parse_ssm_examples(
        "Domanda 1\nTesto?\nA) uno (corretta)\nB) due\nC) tre\nD) quattro\nE) cinque\n"
    )
    batch = LangChainQuestionGenerator(llm).generate(unit, [section], 1, examples, None)
    [question] = batch.questions
    assert question.correct_option_id == "B" and question.difficulty == "difficile"
    assert question.page == 2 and question.options[4].id == "E"
    assert question.options[1].text == "opt B"  # copied answer marker stripped
    # The prompt carries the section text and the few-shot examples.
    messages = llm._model.runnable.invocations[0]  # type: ignore[attr-defined]
    user_prompt = messages[1].content
    assert section.text in user_prompt and "Domanda 1" in user_prompt
    assert "exactly 1 question" in user_prompt


def test_judge_and_style_and_taxonomy_adapters(fixture_document: StructuredDocument) -> None:
    verdict = prompts.WireVerdict(
        answer_supported=False,
        single_correct=True,
        distractors_wrong=True,
        verdict="discard",
        reason="no",
    )
    judge = LangChainQuestionJudge(
        structured([{"raw": FakeAIMessage(), "parsed": verdict, "parsing_error": None}])
    )
    result = judge.judge(good_item(), "passaggio")
    assert result.verdict == "discard" and result.answer_supported is False

    scores = prompts.WireStyleScores(
        ssm_style=5, distractor_quality=4, explanation_clarity=3, groundedness=5, rationale="bene"
    )
    style = LangChainStyleJudge(
        structured([{"raw": FakeAIMessage(), "parsed": scores, "parsing_error": None}])
    )
    assert style.score(good_item(), [], [good_item("g")]).mean == 4.25

    wire_taxonomy = prompts.WireTaxonomy(
        assignments=[
            prompts.WireAssignment(
                section_id="s001", subject=" Cardiologia ", topic="Ipertensione"
            ),
            prompts.WireAssignment(section_id="s002", subject="Cardiologia", topic="Ipertensione"),
            prompts.WireAssignment(section_id="s002", subject="Nefrologia", topic="Duplicato"),
            prompts.WireAssignment(section_id="s003", subject="Cardiologia", topic="Diagnosi"),
        ]
    )
    builder = LangChainTaxonomyBuilder(
        structured([{"raw": FakeAIMessage(), "parsed": wire_taxonomy, "parsing_error": None}])
    )
    taxonomy = builder.build(fixture_document, "r")
    assert [s.name for s in taxonomy.subjects] == ["Cardiologia"]  # duplicate id ignored
    assert [t.name for t in taxonomy.subjects[0].topics] == ["Ipertensione", "Diagnosi"]
    label = taxonomy.label_for("s002")
    assert label is not None and label.topic == "Ipertensione"
    assert taxonomy.section_count == 3
    prompt = prompts.taxonomy_user_prompt(fixture_document)
    assert "id=s002" in prompt and "return exactly 10 assignments" in prompt


class FakeEmbeddingsBackend:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


class BrokenEmbeddingsBackend:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("down")


def test_langchain_embeddings_tracks_usage() -> None:
    tracker = InMemoryUsageTracker("r")
    adapter = LangChainEmbeddings(
        FakeEmbeddingsBackend(), model_name="text-embedding-3-small", tracker=tracker
    )  # type: ignore[arg-type]
    vectors = adapter.embed(["ciao", "mondo bello"])
    assert vectors == [[4.0, 1.0], [11.0, 1.0]]
    assert adapter.embed([]) == []
    [record] = tracker.records()
    assert (
        isinstance(record, UsageRecord) and record.kind == "embedding" and record.input_tokens > 0
    )
    with pytest.raises(LLMError):
        LangChainEmbeddings(BrokenEmbeddingsBackend(), model_name="x").embed(["a"])  # type: ignore[arg-type]


def test_count_tokens_falls_back_for_unknown_model() -> None:
    assert count_tokens(["a" * 40], "unknown-model-xyz") == 10
